#!/usr/bin/env python

"""
A script to train Random Forests on training MC DL1 files. It can train 
both mono and stereo models. Both are trained per telescope as in the stereo 
reconstruction the resulting parameters are just averaged. For stereo reconstruction, 
however, one may use additional features such as h_max or impact_distance as results 
of geometrical reconstruction.
- Inputs are MC DL1 diffuse gamma file and MC DL1 diffuse proton file. These are usualy 
merged from individual DL1 files using sst1mpipe_merge_hdf5 script to reach satisfactory 
statistics for RF training
- Outputs are trained models in scikit.learn format (.sav)

Usage:

$> python sst1mpipe_mc_train_rfs.py
--input-file-gamma gamma_200_300E3GeV_30_30deg_training_dl1.h5
--input-file-proton proton_400_500E3GeV_30_30deg_training_dl1.h5
--output-dir ./
--config sst1mpipe_config.json
--telescope tel_001
--plot-features
--stereo

"""

import sst1mpipe
import argparse
import sys
import os
import logging
import shutil
from sst1mpipe.io import (
    load_dl1_sst1m,
    load_config,
    check_outdir
)
from sst1mpipe.utils import get_telescopes
from sst1mpipe.reco import (
    train_models
)

def parse_args():
    parser = argparse.ArgumentParser(description="MC train RFs")

    # Required arguments
    parser.add_argument(
                        '--input-file-gamma', '--fg',
                        dest='gammas',
                        required=True,
                        help='Path to the DL1 diffuse gamma file for training'
                        )
    parser.add_argument('--config', '-c', action='store', type=str,
                        dest='config_file',
                        help='Path to a configuration file.',
                        required=True
                        )

    # Optional arguments
    parser.add_argument(
                        '--input-file-proton', '--fp',
                        dest='protons',
                        default=None,
                        help='Path to the DL1 diffuse proton file for training',
                        )
    parser.add_argument(
                        '--output-dir', '-o', type=str,
                        dest='outdir',
                        help='Path to store the trained models.',
                        default='./'
                        )

    parser.add_argument(
                        '--telescope', '-t', type=str,
                        dest='telescope',
                        help='Selected telescope: tel_001 or tel_002',
                        default='tel_001'
                        )

    parser.add_argument(
                        '--plot-features',
                        action='store_true',
                        help='Plot feature importances and save figs in the output directory.',
                        dest='plot'
                        )

    parser.add_argument(
                        '--stereo',
                        action='store_true',
                        help='Train RFs for stereo reconstruction.',
                        dest='stereo'
                        )

    parser.add_argument(
                        '--train-stereo-meta-only',
                        action='store_true',
                        help='Train only stage-2 stereo energy meta-regressor (meta-only mode). '
                             'Requires --base-models-dir pointing to pre-trained stage-1 models.',
                        dest='train_stereo_meta_only'
                        )

    parser.add_argument(
                        '--base-models-dir',
                        type=str,
                        dest='base_models_dir',
                        default=None,
                        help='Directory containing pre-trained stage-1 energy models (reg_energy_tel_*.sav) '
                             'for meta-only training mode.',
                        )

    args = parser.parse_args()
    return args

def main():

    args = parse_args()

    input_file_gamma = args.gammas
    input_file_proton = args.protons
    outdir = args.outdir
    telescope = args.telescope
    plot = args.plot
    stereo = args.stereo
    train_stereo_meta_only = args.train_stereo_meta_only
    base_models_dir = args.base_models_dir

    check_outdir(outdir)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers= [
            logging.FileHandler(outdir+'/sst1mpipe_rfs_training_'+telescope+'.log', 'w+'),
            logging.StreamHandler(stream=sys.stdout)
            ]
    )
    logging.info('sst1mpipe version: %s', sst1mpipe.__version__)

    config = load_config(args.config_file)
    output_cfgfile = os.path.join(outdir, input_file_gamma.split('/')[-1].rstrip(".h5") + "_train.cfg")
    shutil.copyfile(args.config_file, output_cfgfile)

    logging.info('Training Random Forests for Telescope %s', telescope)

    params_gamma = load_dl1_sst1m(input_file_gamma, tel=telescope, config=config, table='pandas', check_finite=True, stereo=stereo, quality_cuts=True)
    if input_file_proton is not None:
        params_protons = load_dl1_sst1m(input_file_proton, tel=telescope, config=config, table='pandas', check_finite=True, stereo=stereo, quality_cuts=True)
    else: 
        params_protons = None

    params_gamma_tel1 = None
    params_gamma_tel2 = None
    telescope_pair = None
    train_stereo_meta_this_run = False
    stereo_meta_training_mode = 'full_oof'
    stereo_meta_base_models_dir = None
    
    stereo_meta_enabled = config.get('stereo_energy_meta', {}).get('enabled', False) or train_stereo_meta_only
    if stereo and stereo_meta_enabled:
        if train_stereo_meta_only and base_models_dir is None:
            raise ValueError('--train-stereo-meta-only requires --base-models-dir')
        all_tels = sorted(get_telescopes(input_file_gamma), key=lambda t: int(t.split('_')[-1]))
        if len(all_tels) >= 2:
            telescope_pair = (all_tels[0], all_tels[1])
            
            # Deterministic: only train meta on primary (first sorted) telescope run
            is_primary_telescope = telescope == telescope_pair[0]
            
            # Determine training mode: CLI flag overrides config
            if train_stereo_meta_only:
                stereo_meta_training_mode = 'meta_only'
                stereo_meta_base_models_dir = base_models_dir
                train_stereo_meta_this_run = is_primary_telescope
            else:
                stereo_meta_training_mode = config.get('stereo_energy_meta', {}).get('training_mode', 'full_oof')
                train_stereo_meta_this_run = is_primary_telescope
            
            if train_stereo_meta_this_run:
                logging.info(
                    'Loading stereo gamma tables for meta-regressor training. '
                    'Telescope pair: %s. Primary telescope (this run): %s. Mode: %s',
                    telescope_pair,
                    telescope,
                    stereo_meta_training_mode,
                )
                params_gamma_tel1 = load_dl1_sst1m(
                    input_file_gamma,
                    tel=telescope_pair[0],
                    config=config,
                    table='pandas',
                    check_finite=True,
                    stereo=True,
                    quality_cuts=True,
                )
                params_gamma_tel2 = load_dl1_sst1m(
                    input_file_gamma,
                    tel=telescope_pair[1],
                    config=config,
                    table='pandas',
                    check_finite=True,
                    stereo=True,
                    quality_cuts=True,
                )
            else:
                logging.info(
                    'Stereo meta training skipped for this run (not primary telescope). '
                    'Telescope: %s. Primary: %s.',
                    telescope,
                    telescope_pair[0],
                )
        else:
            logging.warning('Stereo meta enabled, but less than two telescopes found in gamma file.')

    train_models(
        params_gamma,
        params_protons,
        config=config,
        plot=plot,
        outdir=outdir,
        telescope=telescope,
        stereo=stereo,
        params_gamma_tel1=params_gamma_tel1,
        params_gamma_tel2=params_gamma_tel2,
        telescope_pair=telescope_pair,
        train_stereo_meta_this_run=train_stereo_meta_this_run,
        stereo_meta_training_mode=stereo_meta_training_mode,
        stereo_meta_base_models_dir=stereo_meta_base_models_dir,
    )
    
if __name__ == '__main__':
    main()