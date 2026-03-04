#!/usr/bin/env python

"""
A script to train Random Forests on training MC DL1 files. It can train 
both mono and stereo models. 

STEREO MODES:
- Standard Stereo (--stereo): Trains per-telescope energy regressors, but includes stereo features (h_max, impact_distance)
- True Stereo (--stereo --true-stereo): Trains a SINGLE energy regressor using combined features from BOTH telescopes

Both are trained with disp_norm reconstruction (no disp_sign in stereo). For true stereo reconstruction, 
a single energy regressor is trained using features from both telescopes simultaneously, enabling better
multi-telescope energy estimation compared to averaging individual telescope predictions.

Inputs are MC DL1 diffuse gamma file and MC DL1 diffuse proton file. These are usually 
merged from individual DL1 files using sst1mpipe_merge_hdf5 script to reach satisfactory
statistics for RF training.

Outputs are trained models in scikit-learn format (.sav).

Usage:

# Standard monocular training:
$> python sst1mpipe_mc_train_rfs.py \\
   --input-file-gamma gamma_30deg_training_dl1.h5 \\
   --input-file-proton proton_30deg_training_dl1.h5 \\
   --output-dir ./models_mono/ \\
   --config sst1mpipe_config.json \\
   --telescope tel_001 \\
   --plot-features

# Standard stereo training (per-telescope):
$> python sst1mpipe_mc_train_rfs.py \\
   --input-file-gamma gamma_30deg_training_dl1.h5 \\
   --input-file-proton proton_30deg_training_dl1.h5 \\
   --output-dir ./models_stereo/ \\
   --config sst1mpipe_config.json \\
   --telescope tel_001 \\
   --stereo \\
   --plot-features

# TRUE STEREO training (combined single regressor):
$> python sst1mpipe_mc_train_rfs.py \\
   --input-file-gamma gamma_30deg_training_dl1.h5 \\
   --input-file-proton proton_30deg_training_dl1.h5 \\
   --output-dir ./models_true_stereo/ \\
   --config sst1mpipe_config.json \\
   --telescope tel_001 \\
   --stereo --true-stereo \\
   --plot-features

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
                        '--true-stereo',
                        action='store_true',
                        help='Train TRUE STEREO energy regressor using combined features from both telescopes. '
                             'This trains a single energy regressor on data from both tel_001 and tel_002, '
                             'instead of training separate regressors per-telescope.',
                        dest='true_stereo'
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
    true_stereo = args.true_stereo

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
    
    # Check for conflicting options
    if true_stereo and not stereo:
        logging.warning('--true-stereo flag set but --stereo flag is not set.')
        logging.warning('Enabling --stereo automatically for compatibility...')
        stereo = True

    params_gamma = load_dl1_sst1m(input_file_gamma, tel=telescope, config=config, table='pandas', check_finite=True, stereo=stereo, quality_cuts=True)
    if input_file_proton is not None:
        params_protons = load_dl1_sst1m(input_file_proton, tel=telescope, config=config, table='pandas', check_finite=True, stereo=stereo, quality_cuts=True)
    else: 
        params_protons = None

    # For true stereo, also load data from the other telescope for energy training
    params_gamma_tel2 = None
    params_protons_tel2 = None
    if true_stereo:
        other_telescope = 'tel_002' if telescope == 'tel_001' else 'tel_001'
        logging.info('Loading data from %s for TRUE STEREO energy training...', other_telescope)
        params_gamma_tel2 = load_dl1_sst1m(input_file_gamma, tel=other_telescope, config=config, table='pandas', check_finite=True, stereo=stereo, quality_cuts=True)
        if input_file_proton is not None:
            params_protons_tel2 = load_dl1_sst1m(input_file_proton, tel=other_telescope, config=config, table='pandas', check_finite=True, stereo=stereo, quality_cuts=True)

    train_models(params_gamma, params_protons, config=config, plot=plot, outdir=outdir, telescope=telescope, stereo=stereo, 
                 true_stereo=true_stereo, params_gamma_tel2=params_gamma_tel2, params_protons_tel2=params_protons_tel2)
    
if __name__ == '__main__':
    main()