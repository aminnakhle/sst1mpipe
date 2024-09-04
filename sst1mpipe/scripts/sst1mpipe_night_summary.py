#!/usr/bin/env python

"""
A script to create night summary for each observed source, which is 
stored as a PDF in the base data analysis directory. For the full
summary, the script needs all data analysis levels up to DL3 for 
mono and stereo to be already produced. It also needs the DL1 
distributions to be already extracted.
- Input is a base data analysis directory, within which the expected 
structure is BASE_DIR/{date}/{source}/{telescope}/{data level}/{subdir}/
- Outputs are night summary PDFs for all sources observed during the night

Usage:

$> python sst1mpipe_night_summary.py
--base-dir /data/
--out-dir /outputs/
--date 20240810
--config sst1mpipe_config.json
--sub-dir /v0.5.5/

"""

import argparse
import sys
import os
import logging

import sst1mpipe

from sst1mpipe.io import (
    load_dl1_sst1m,
    load_dl2_sst1m,
    load_config,
    check_outdir
)
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sst1mpipe.utils import (
    get_moon_params,
)
import astropy.units as u
from gammapy.data import DataStore
from ctapipe.io import read_table
from astropy.time import Time
from sst1mpipe.utils.NSB_tools import plot_average_nsb_VS_time
from astropy.table import vstack
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Makes run summary PDF for all sources observed during a single night.")

    parser.add_argument(
                        '--base-dir', '-d',
                        dest='base_dir',
                        required=True,
                        type=str,
                        help='Base data analysis directory.'
                        )
    parser.add_argument(
                        '--out-dir', '-o',
                        dest='out_dir',
                        type=str,
                        help='Output directory to store the night summary.',
                        default=''
                        )

    parser.add_argument(
                        '--date',
                        dest='date',
                        default=None,
                        type=str,
                        help='Date of the observing night (YYYYMMDD).',
                        required=True
                        )

    parser.add_argument('--config', '-c', action='store', type=str,
                        dest='config_file',
                        help='Path to a configuration file.',
                        required=True
                        )

    parser.add_argument(
                        '--sub-dir', '-s', type=str,
                        dest='version',
                        help='Sub directory for given data sample inside each data level directory, e.g. a version of sst1mpipe used (../DL{1,2,3}/v0.5.5/).',
                        default=''
                        )

    args = parser.parse_args()
    return args


def get_sources(base_path):
    sources = []
    directories=[d for d in os.listdir(base_path) if os.path.isdir(base_path + '/'+ d)]
    print(directories)
    for d in directories:
        if d not in ['log', 'output', 'error', 'UNKNOWN', 'dark', 'DARK', 'drak', 'DRAK']:
            sources.append(d)
    return sources


def load_dl1_pedestals(input_file):
    
    pedestals = read_table(input_file, "/dl1/monitoring/telescope/pedestal")
    return pedestals


def load_files(files, config=None, tel=None, level='dl1', stereo=False):
    i = 0
    ped_table = []
    data = []

    for input_file in files:
        try:
            if level == 'dl1':
                df = load_dl1_sst1m(input_file, tel=tel, config=config, table='pandas', stereo=stereo)
                if not stereo:
                    try:
                        pt = load_dl1_pedestals(input_file)
                    except: 
                        pass
            elif level == 'dl2':
                df = load_dl2_sst1m(input_file, tel=tel, config=config, table='pandas')
        except:
            print("Broken file", input_file)
            continue
        if i == 0:
            data = df
            if (level == 'dl1') and not stereo:
                try:
                    ped_table = pt
                except:
                    pass
        else:
            try:
                data = pd.concat([data, df])
            except:
                print("Broken file", input_file)

            if (level == 'dl1') and not stereo:
                try:
                    ped_table = vstack([ped_table, pt])
                except:
                    print("No pedestal monitoring in file", input_file)
            
        i += 1
    return data, ped_table


def make_file_list(path, stereo=False, level='dl1'):
    
    files_selected = []
    
    if stereo:
        if level == 'dl1':
            files = glob.glob(path + '/*dl1_stereo.h5')
        else:
            files = glob.glob(path + '/*stereo_dl2.h5')
    else:
        files = glob.glob(path + '/*'+level+'.h5')

    files_selected = files_selected + files
    
    files_selected = np.sort(files_selected)
    return files_selected


def prepare_to_pdf(image_path):

    img1 = Image.open(image_path)
    img1.load() # required for png.split()
    background1 = Image.new("RGB", img1.size, (255, 255, 255))
    background1.paste(img1, mask=img1.split()[3])

    return background1


def get_distributions(dist_path=None, data_store=None):

    histograms = []
    histograms_diff = []
    zeniths = []
    obsids_sorted = []
    livetimes = []
    survived_ped = []

    no_ped = 0
    tables =  glob.glob(dist_path+'/intensity*.h5')
    if len(tables) == 0:
        # this is to read the data from date when we do not have pedestal events, so no recleaning, but we still index them in dl3 index files
        tables =  glob.glob(dist_path+'/intensity*.h5')
        no_ped = 1

    for table in tables:
        obsid = table.split('/')[-1].split('.')[0].split('_')[-1]
        hist = read_table(table, 'intensity_hist')
        t_elapsed = read_table(table, 't_elapsed')
        try:
            zenith = read_table(table, 'zenith')
        except:
            zenith = read_table(table, 'z_elapsed')

        # so that the pedestal fraction cut does not remove data for which we do not have pedestals (in those distribution files there is pedestal_frac=100.)
        if no_ped:
            survived_pedestal_frac = [0.]
        else:
            survived_pedestal_frac = read_table(table, 'survived_pedestal_frac')

        mask_datastore = data_store.obs_table['OBS_ID'] == int(obsid)
        if sum(mask_datastore) == 0:
            print(int(obsid),'not in datastore!')
            continue

        histograms.append(np.array(hist['rate']).astype(np.float64))
        histograms_diff.append(np.array(hist['diff_rate']).astype(np.float64))
        zeniths.append(np.array(zenith)[0].astype(np.float64))
        obsids_sorted.append(int(obsid))
        survived_ped.append(np.array(survived_pedestal_frac)[0].astype(np.float64))
        bins = np.hstack((np.array(hist['low']),hist['high'][-1]))    
        mask_datastore = data_store.obs_table['OBS_ID'] == int(obsid)
        livetime = data_store.obs_table['LIVETIME'].value
        livetimes.append(livetime[mask_datastore][0])
    
    histograms = np.array(histograms)
    histograms_diff = np.array(histograms_diff)
    zeniths = np.array(zeniths)
    survived_ped = np.array(survived_ped)
    obsids_sorted = np.array(obsids_sorted)
    livetimes = np.array(livetimes).flatten()

    return histograms, histograms_diff, zeniths, obsids_sorted, livetimes, survived_ped, bins


def get_min_max_times(dl1_files, tel=None, config=None, stereo=False):

    print(dl1_files[0])
    print(dl1_files[-1])
    dl1_files = np.sort(dl1_files)
    ind = 0
    first_loaded = False
    while not first_loaded:
        try:
            df = load_dl1_sst1m(dl1_files[ind], tel=tel, config=config, table='pandas', stereo=stereo)
            first_loaded = True
        except:
            print('Broken file')
            ind += 1
            if ind > 50: 
                first_loaded = True
                print('Something is really wrong, too many broken files!')
    min_time = min(df['local_time'])
    ind = 0
    last_loaded = False
    while not last_loaded:
        try:
            df = load_dl1_sst1m(dl1_files[len(dl1_files)-ind], tel=tel, config=config, table='pandas', stereo=stereo)
            last_loaded = True
        except:
            print('Broken file')
            ind += 1
            if ind > 50: 
                last_loaded = True
                print('Something is really wrong, too many broken files!')
    max_time = max(df['local_time'])
    return min_time, max_time


def main():

    args = parse_args()

    base_path = args.base_dir
    date = args.date
    config_file = args.config_file
    version = args.version
    out_dir = args.out_dir
    if len(out_dir) > 0:
        outpath = out_dir
    else:
        outpath = base_path + '/' + date + '/'
        print('Default outpath used: %s', outpath)

    check_outdir(outpath)

    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers= [
                logging.FileHandler(outpath+'/sst1mpipe_night_summary_'+str(sst1mpipe.__version__) + '.log', 'w+'),
                logging.StreamHandler(stream=sys.stdout)
                ]
        )
    logging.info('sst1mpipe version: %s', sst1mpipe.__version__)

    sources = get_sources(base_path + '/' + date + '/')

    config = load_config(config_file)

    tels = ['cs1', 'cs2', 'stereo']

    for source in sources:

        fig, ax = plt.subplots(1, 1, figsize=(12, 5)) # trigger rates
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 5)) # trigger rates (zoom)
        fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5)) # dl2 event rates
        fig3, ax3 = plt.subplots(1, 1, figsize=(12, 5)) # NSB
        fig4, ax4 = plt.subplots(1, 2, figsize=(12,5)) # distributions, tel1
        fig5, ax5 = plt.subplots(1, 2, figsize=(12,5)) # distributions, tel2
        fig6, ax6 = plt.subplots(1, 2, figsize=(12,5)) # distributions, stereo
        fig7, ax7 = plt.subplots(1, 2, figsize=(12, 5)) # zenith angles
        fig8, ax8 = plt.subplots(1, 2, figsize=(12, 5)) # cogs, dl1 mono, survived cleaning
        fig9, ax9 = plt.subplots(1, 2, figsize=(12, 5)) # cogs, dl2 mono, photons
        fig10, ax10 = plt.subplots(1, 2, figsize=(12, 5)) # DL1 stereo, survived cleaning
        fig11, ax11 = plt.subplots(1, 1, figsize=(6, 5)) # Moon

        median1 = []
        median2 = []

        rate_colors = {'tel_021':'blue', 'tel_022':'orange', 'stereo':'green'}

        # making DL1 time bins
        bunch_size = 20 # N of files
        for telescope in ['cs1', 'cs2']:
            base_path_source = base_path + '/' + date + '/' + source+'/' + telescope
            dl1_path = base_path_source + '/DL1/' + version
            is_dl1 = len(glob.glob(dl1_path+'/'+'*.h5'))
            # handle lower case directories
            if not is_dl1:
                dl1_path = base_path_source + '/dl1/' + version
                is_dl1 = len(glob.glob(dl1_path+'/'+'*.h5'))

            if telescope == 'cs1':
                if is_dl1:
                    dl1_files = make_file_list(dl1_path, stereo=False, level='dl1')
                    tel1_min, tel1_max = get_min_max_times(dl1_files, tel='tel_021', config=config)
                else:
                    tel1_min = 1e15
                    tel1_max = 0
            elif telescope == 'cs2':
                if is_dl1:
                    dl1_files = make_file_list(dl1_path, stereo=False, level='dl1')
                    tel2_min, tel2_max = get_min_max_times(dl1_files, tel='tel_022', config=config)
                else:
                    tel2_min = 1e15
                    tel2_max = 0
    
        min_time_dl1 = min([tel1_min, tel2_min])
        max_time_dl1 = max([tel1_max, tel2_max])
        dl1_rate_bins = np.linspace(min_time_dl1, max_time_dl1, int((max_time_dl1-min_time_dl1)/10.))

        for telescope in tels:

            # Load all files
            base_path_source = base_path + '/' + date + '/' + source+'/' + telescope
            dl1_path = base_path_source + '/DL1/' + version
            dl2_path = base_path_source + '/DL2/' + version
            dl3_path = base_path_source + '/DL3/' + version
            dist_path = base_path_source + '/distributions/' + version
            
            is_dl1 = len(glob.glob(dl1_path+'/'+'*.h5'))
            is_dl2 = len(glob.glob(dl2_path+'/'+'*.h5'))
            is_dl3 = len(glob.glob(dl3_path+'/'+'*.fits'))
            is_dist = len(glob.glob(dist_path+'/'+'*.h5'))

            # handle low case directories
            if not is_dl1:
                dl1_path = base_path_source + '/dl1/' + version
                is_dl1 = len(glob.glob(dl1_path+'/'+'*.h5'))
            if not is_dl2:
                dl2_path = base_path_source + '/dl2/' + version
                is_dl2 = len(glob.glob(dl2_path+'/'+'*.h5'))
            if not dl3_path:
                dl3_path = base_path_source + '/dl3/' + version
                is_dl3 = len(glob.glob(dl3_path+'/'+'*.fits'))

            if telescope not in 'stereo':
                stereo=False
            else:
                stereo=True
            
            if is_dl1:
                dl1_files = make_file_list(dl1_path, stereo=stereo, level='dl1')
            if is_dl2:
                dl2_files = make_file_list(dl2_path, stereo=stereo, level='dl2')
            
            if '1' in telescope:
                tel = 'tel_021'
                tt = 21
            elif '2' in telescope:
                tel = 'tel_022'
                tt = 22
            else:
                tel = 'stereo'
                tt = 0

            if is_dl1:
                # We load and plot DL1 rates in bunches of file, 
                # because otherwise it needs too much memory and
                # is realy slow.
                N_bunches = int(len(dl1_files)/bunch_size)
                h11_tot = np.zeros((100, 100))
                h22_tot = np.zeros((100, 100))
                if tel != 'stereo':
                    h_tot = np.zeros(len(dl1_rate_bins)-1)
                    for i in range(N_bunches):
                        print('BUNCH', i)
                        dl1, ped_table = load_files(dl1_files[i*bunch_size:bunch_size*(i+1)], tel=tel, level='dl1')
                        # Trigger rates
                        if (len(dl1) > 0) and (len(ped_table) > 0):
                            h1 = np.histogram(dl1.local_time, bins=dl1_rate_bins)
                            h_tot += h1[0]
                            # pedestal table
                            plot_average_nsb_VS_time(ped_table,tt,ax=ax3, color=rate_colors[tel])
                            # CoGs - DL1 mono, survived cleaning
                            if tel == 'tel_021':
                                h11, xedges, yedges = np.histogram2d(dl1['camera_frame_hillas_x'].dropna(), dl1['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                                h11_tot += h11
                            else:
                                h22, xedges, yedges = np.histogram2d(dl1['camera_frame_hillas_x'].dropna(), dl1['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                                h22_tot += h22
                    # last bunch
                    dl1, ped_table = load_files(dl1_files[bunch_size*(i+1):], tel=tel, level='dl1')
                    if len(dl1) > 0:
                        h1 = np.histogram(dl1.local_time, bins=dl1_rate_bins)
                        h_tot += h1[0]
                    centers = (dl1_rate_bins[1:]+dl1_rate_bins[:-1]) / 2
                    ax.plot(centers, h_tot/10, label=tel, alpha=0.7, color=rate_colors[tel])
                    ax1.plot(centers, h_tot/10, label=tel, alpha=0.7, color=rate_colors[tel])
                    median1.append(np.median(h_tot/10.))
                    # pedestal table
                    if len(ped_table) > 0:
                        plot_average_nsb_VS_time(ped_table,tt,ax=ax3, color=rate_colors[tel], label=tel)
                        # CoGs - DL1 mono, survived cleaning
                    X, Y = np.meshgrid(xedges, yedges)
                    if tel == 'tel_021':
                        if len(dl1) > 0:
                            h11, xedges, yedges = np.histogram2d(dl1['camera_frame_hillas_x'].dropna(), dl1['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                            h11_tot += h11
                        ax8[0].pcolormesh(X, Y, h11_tot)
                    else:
                        if len(dl1) > 0:
                            h22, xedges, yedges = np.histogram2d(dl1['camera_frame_hillas_x'].dropna(), dl1['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                            h22_tot += h22
                        ax8[1].pcolormesh(X, Y, h22_tot)
                else:
                    h_tot = np.zeros(len(dl1_rate_bins)-1)
                    for i in range(N_bunches):
                        dl1_stereo_1,_ = load_files(dl1_files[i*bunch_size:bunch_size*(i+1)], tel='tel_021', level='dl1', stereo=stereo)
                        dl1_stereo_2,_ = load_files(dl1_files[i*bunch_size:bunch_size*(i+1)], tel='tel_022', level='dl1', stereo=stereo)
                        # Trigger rates
                        if (len(dl1_stereo_1) > 0) and (len(dl1_stereo_2) > 0):
                            h1 = np.histogram(dl1_stereo_1.local_time, bins=dl1_rate_bins)
                            h_tot += h1[0]
                            # CoG - DL1 stereo, survived cleaning
                            h11, xedges, yedges = np.histogram2d(dl1_stereo_1['camera_frame_hillas_x'].dropna(), dl1_stereo_1['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                            h22, xedges, yedges = np.histogram2d(dl1_stereo_2['camera_frame_hillas_x'].dropna(), dl1_stereo_2['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                            h11_tot += h11
                            h22_tot += h22

                    # last bunch
                    dl1_stereo_1,_ = load_files(dl1_files[bunch_size*(i+1):], tel='tel_021', level='dl1', stereo=stereo)
                    dl1_stereo_2,_ = load_files(dl1_files[bunch_size*(i+1):], tel='tel_022', level='dl1', stereo=stereo)
                    if (len(dl1_stereo_1) > 0) and (len(dl1_stereo_2) > 0):
                        h1 = np.histogram(dl1_stereo_1.local_time, bins=dl1_rate_bins)
                        h_tot += h1[0]
                        h11, xedges, yedges = np.histogram2d(dl1_stereo_1['camera_frame_hillas_x'].dropna(), dl1_stereo_1['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                        h22, xedges, yedges = np.histogram2d(dl1_stereo_2['camera_frame_hillas_x'].dropna(), dl1_stereo_2['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                        h11_tot += h11
                        h22_tot += h22
                    centers = (dl1_rate_bins[1:]+dl1_rate_bins[:-1]) / 2
                    ax.plot(centers, h_tot/10, label=tel, alpha=0.7, color=rate_colors[tel])
                    ax1.plot(centers, h_tot/10, label=tel, alpha=0.7, color=rate_colors[tel])
                    median1.append(np.median(h_tot/10.))
                    X, Y = np.meshgrid(xedges, yedges)
                    ax10[0].pcolormesh(X, Y, h11_tot)
                    ax10[1].pcolormesh(X, Y, h22_tot)

            if is_dl2:
                # Reconstructed event rates (zoomed)
                h_tot = np.zeros(len(dl1_rate_bins)-1)
                h11_tot = np.zeros((100, 100))
                h22_tot = np.zeros((100, 100))
                zeniths_all = np.zeros(50)
                local_time = []
                zenith_time = []
                time_all = []
                moon_altaz_all = []
                moon_separation_all = []
                moon_phase_angle_all = []
                N_bunches = int(len(dl2_files)/bunch_size)
                for i in range(N_bunches):
                    print('BUNCH', i)
                    dl2,_ = load_files(dl2_files[i*bunch_size:bunch_size*(i+1)], tel=tel, level='dl2')
                    if len(dl2) > 0:
                        h1 = np.histogram(dl2.local_time, bins=dl1_rate_bins)
                        h_tot += h1[0]

                    # Zenith angles
                    if (tel != 'stereo') and (len(dl2) > 0):
                        zenith = 90-dl2['true_alt_tel']
                        h1 = np.histogram(zenith, bins=50, range=[0, 65])
                        zeniths_all += h1[0]
                        local_time.append(np.array(dl2['local_time'].T))
                        zenith_time.append(np.array(zenith.T))

                    # moon
                    if (tel == 'tel_021') and (len(dl2) > 0):
                        time, moon_altaz, moon_separation, moon_phase_angle = get_moon_params(dl2, config=config, tel=tel, thinning=100)
                        time_all.append(np.array(time.unix))
                        moon_altaz_all.append(np.array(moon_altaz.alt.to_value(u.deg)))
                        moon_separation_all.append(np.array(moon_separation.to_value(u.deg)))
                        moon_phase_angle_all.append(np.array(moon_phase_angle.to_value(u.deg)))

                    # CoGs - DL2 mono, gammas
                    if len(dl2) > 0:
                        mask1 = dl2['gammaness'] > 0.7
                        if tel == 'tel_021':
                            h11, xedges, yedges = np.histogram2d(dl2[mask1]['camera_frame_hillas_x'].dropna(), dl2[mask1]['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                            h11_tot += h11
                        elif tel == 'tel_022':
                            h22, xedges, yedges = np.histogram2d(dl2[mask1]['camera_frame_hillas_x'].dropna(), dl2[mask1]['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                            h22_tot += h22

                # last bunch
                dl2,_ = load_files(dl2_files[bunch_size*(i+1):], tel=tel, level='dl2')
                if len(dl2) > 0:
                    h1 = np.histogram(dl2.local_time, bins=dl1_rate_bins)
                    h_tot += h1[0]
                centers = (dl1_rate_bins[1:]+dl1_rate_bins[:-1]) / 2
                ax2.plot(centers, h_tot/10., alpha=0.7, label=tel, color=rate_colors[tel])
                median2.append(np.median(h_tot/10.))
                if tel != 'stereo':
                    if len(dl2) > 0:
                        zenith = 90-dl2['true_alt_tel']
                        h1 = np.histogram(zenith, bins=50, range=[0, 65])
                        zeniths_all += h1[0]
                        local_time.append(np.array(dl2['local_time'].T))
                        zenith_time.append(np.array(zenith.T))
                    ax7[0].step(np.linspace(0, 65, 50), zeniths_all, alpha=0.5, label=tel, where='mid')
                    local_time = np.array(local_time)
                    zenith_time = np.array(zenith_time)
                    ax7[1].plot(np.concatenate(local_time), np.concatenate(zenith_time), '.', label=tel, alpha=0.7)
                if tel == 'tel_021':
                    if len(dl2) > 0:
                        time, moon_altaz, moon_separation, moon_phase_angle = get_moon_params(dl2, config=config, tel=tel, thinning=100)
                        time_all.append(np.array(time.unix))
                        moon_altaz_all.append(np.array(moon_altaz.alt.to_value(u.deg)))
                        moon_separation_all.append(np.array(moon_separation.to_value(u.deg)))
                        moon_phase_angle_all.append(np.array(moon_phase_angle.to_value(u.deg)))

                    ax11.plot(np.concatenate(time_all), np.concatenate(moon_altaz_all), label='Moon alt')
                    ax11.plot(np.concatenate(time_all), np.concatenate(moon_separation_all), label='Moon sep')
                    ax11.plot(np.concatenate(time_all), np.concatenate(moon_phase_angle_all), label='Moon phase (full=0)')
                # CoGs - DL2 mono, gammas
                X, Y = np.meshgrid(xedges, yedges)
                if len(dl2) > 0:
                    mask1 = dl2['gammaness'] > 0.7
                if tel == 'tel_021':
                    if len(dl2) > 0:
                        h11, xedges, yedges = np.histogram2d(dl2[mask1]['camera_frame_hillas_x'].dropna(), dl2[mask1]['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                        h11_tot += h11
                    ax9[0].pcolormesh(X, Y, h11_tot)
                elif tel == 'tel_022':
                    if len(dl2) > 0:
                        h22, xedges, yedges = np.histogram2d(dl2[mask1]['camera_frame_hillas_x'].dropna(), dl2[mask1]['camera_frame_hillas_y'].dropna(), bins=100, range=[[-0.5, 0.5], [-0.5, 0.5]])
                        h22_tot += h22
                    ax9[1].pcolormesh(X, Y, h22_tot)

            # DL3 Datastore
            if is_dl3:
                data_store = DataStore.from_dir(dl3_path)
            if is_dist:
                histograms, histograms_diff, _, _, livetimes, survived_ped, bins = get_distributions(dist_path=dist_path, data_store=data_store)

            # PLOTTING

            # Distributions
            if is_dist and (tt==21):
                p = ax4[0].plot(bins[:-1], histograms_diff.T, alpha=0.5)
                ax4[0].set_xscale('log')
                ax4[0].set_yscale('log')
                ax4[0].axvline(50)
                ax4[0].set_ylabel('Differential rate')
                ax4[0].set_xlabel('Hillas Intensity [p.e.]')
                ax4[0].set_xlim([10, 10**5])
                ax4[0].grid()
                ax4[0].set_title('Differential rates of DL1 events (per wobble)')

                h = ax4[1].hist(livetimes, bins=20)
                ax4[1].grid()
                ax4[1].set_xlabel('livetime [s]')
                ax4[1].set_title('Livetime of each wobble')
                fig4.suptitle(tel, fontsize=16)

            if is_dist and (tt==22):
                p = ax5[0].plot(bins[:-1], histograms_diff.T, alpha=0.5)
                ax5[0].set_xscale('log')
                ax5[0].set_yscale('log')
                ax5[0].axvline(50)
                ax5[0].set_ylabel('Differential rate')
                ax5[0].set_xlabel('Hillas Intensity [p.e.]')
                ax5[0].set_xlim([10, 10**5])
                ax5[0].grid()
                ax5[0].set_title('Differential rates of DL1 events (per wobble)')

                h = ax5[1].hist(livetimes, bins=20)
                ax5[1].grid()
                ax5[1].set_xlabel('livetime [s]')
                ax5[1].set_title('Livetime of each wobble')
                fig5.suptitle(tel, fontsize=16)

            if is_dist and (tt==0):
                p = ax6[0].plot(bins[:-1], histograms_diff.T, alpha=0.5)
                ax6[0].set_xscale('log')
                ax6[0].set_yscale('log')
                ax6[0].axvline(50)
                ax6[0].set_ylabel('Differential rate')
                ax6[0].set_xlabel('Hillas Intensity [p.e.]')
                ax6[0].set_xlim([10, 10**5])
                ax6[0].grid()
                ax6[0].set_title('Differential rates of DL1 events (per wobble)')

                h = ax6[1].hist(livetimes, bins=20)
                ax6[1].grid()
                ax6[1].set_xlabel('livetime [s]')
                ax6[1].set_title('Livetime of each wobble')
                fig6.suptitle(tel, fontsize=16)

            # Zenith angles
            if is_dl2 and (tel != "stereo"):
                fig7.suptitle("Distribution of zenith angles", fontsize=16)
                ax7[0].set_xlabel('zenith [deg]')
                ax7[0].grid()
                ax7[0].set_ylabel('N reconstructed events')
                ax7[0].grid()
                ax7[0].legend()
                ax7[1].set_xlabel('local_time [s]')
                ax7[1].set_ylabel('zenith [deg]')
                ax7[1].grid()
                ax7[1].invert_yaxis()
                ax7[1].legend()

            # CoGs
            # DL1 mono, survived cleaning
            fig8.suptitle("CoG of DL1 mono events", fontsize=16)
            if is_dl1 and (tt == 21):
                ax8[0].set_title('tel_021')
            if is_dl1 and (tt == 22):
                ax8[1].set_title('tel_022')
            
            # DL2 mono
            # Gammas
            fig9.suptitle("CoG of DL2 mono events, gammaness > 0.7", fontsize=16)
            if is_dl2 and (tt == 21):
                ax9[0].set_title(tel)
            if is_dl2 and (tt == 22):
                ax9[1].set_title(tel)

            # DL1 stereo, survived cleaning
            if is_dl1 and (tt == 0):
                fig10.suptitle("CoG of DL1 stereo events", fontsize=16)
                ax10[0].set_title('tel1')
                ax10[1].set_title('tel2')

            # Moon
            if is_dl2 and (tt == 21):
                ax11.grid()
                ax11.set_ylabel('deg')
                ax11.set_xlabel('local_time [s]')
                ax11.legend()
                ax11.set_title('Moon')  


        # changing tick labels to UTC time
        if is_dl1:
            new_times = []
            for i in range(len(ax.get_xticklabels())):
                time_unix = ax.get_xticklabels()[i].get_position()[0]
                new_times.append(Time(time_unix, format='unix', scale='utc').isot.split('T')[1].split('.')[0])
            t = ax.set_xticklabels(new_times) #, rotation='vertical')
            ax.set_xlabel('Time [UTC]')
            ax.set_ylabel('Trigger rate [Hz]')
            ax.grid()
            ax.legend()
            ax.set_title('Trigger rates', fontsize=16)
        
        # changing tick labels to UTC time
        if is_dl1:
            new_times = []
            for i in range(len(ax1.get_xticklabels())):
                time_unix = ax1.get_xticklabels()[i].get_position()[0]
                new_times.append(Time(time_unix, format='unix', scale='utc').isot.split('T')[1].split('.')[0])
            t = ax1.set_xticklabels(new_times) #, rotation='vertical')
            ax1.set_xlabel('Time [UTC]')
            ax1.set_ylabel('Trigger rate [Hz]')
            ax1.grid()
            ax1.legend()
            ax1.set_ylim([0, 2*max(median1)])
            ax1.set_title('Trigger rates (zoom)', fontsize=16)

        # changing tick labels to UTC time
        if is_dl2:
            new_times = []
            for i in range(len(ax2.get_xticklabels())):
                time_unix = ax2.get_xticklabels()[i].get_position()[0]
                new_times.append(Time(time_unix, format='unix', scale='utc').isot.split('T')[1].split('.')[0])
            t = ax2.set_xticklabels(new_times) #, rotation='vertical')
            ax2.set_xlabel('Time [UTC]')
            ax2.set_ylabel('Event rate [Hz]')
            ax2.set_ylim([0, 2*max(median2)])
            ax2.grid()
            ax2.legend()
            ax2.set_title('Rates of reconstructed events', fontsize=16)

        if is_dl1:
            ax3.grid()
            ax3.legend()
            ax3.set_xlabel('Time [UTC]')
            ax3.set_ylabel('NSB rate [MHz]')
            ax3.set_title("Night Sky Background", fontsize=16)
    
        fig.savefig(outpath+'/trigger_rates.png', dpi=250)
        fig1.savefig(outpath+'/trigger_rates_zoom.png', dpi=250)
        fig2.savefig(outpath+'/reco_rates_zoom.png', dpi=250)
        fig3.savefig(outpath+'/nsb.png', dpi=250)
        fig4.savefig(outpath+'/diff_rates_1.png', dpi=250)
        fig5.savefig(outpath+'/diff_rates_2.png', dpi=250)
        fig6.savefig(outpath+'/diff_rates_stereo.png', dpi=250)
        fig7.savefig(outpath+'/zenith_angles.png', dpi=250)
        fig8.savefig(outpath+'/cog_dl1_mono.png', dpi=250)
        fig9.savefig(outpath+'/cog_dl2_mono_photons.png', dpi=250)
        fig10.savefig(outpath+'/cog_dl1_stereo.png', dpi=250)
        fig11.savefig(outpath+'/moon.png', dpi=250)
    

        # store pdf
        img1 = prepare_to_pdf(outpath+'/trigger_rates.png')
        img2 = prepare_to_pdf(outpath+'/trigger_rates_zoom.png')
        img3 = prepare_to_pdf(outpath+'/reco_rates_zoom.png')
        img4 = prepare_to_pdf(outpath+'/nsb.png')
        img5 = prepare_to_pdf(outpath+'/diff_rates_1.png')
        img6 = prepare_to_pdf(outpath+'/diff_rates_2.png')
        img7 = prepare_to_pdf(outpath+'/diff_rates_stereo.png')
        img8 = prepare_to_pdf(outpath+'/zenith_angles.png')
        img9 = prepare_to_pdf(outpath+'/cog_dl1_mono.png')
        img10 = prepare_to_pdf(outpath+'/cog_dl2_mono_photons.png')
        img11 = prepare_to_pdf(outpath+'/cog_dl1_stereo.png')
        img12 = prepare_to_pdf(outpath+'/moon.png')
        image_list = [img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12]

        pdf_file = 'night_summary_' + str(date) + '_' +source+'.pdf'
        img1.save(outpath + '/' + pdf_file, "PDF", save_all=True, resolution=100.0, append_images=image_list)

        # remove all temporary png images
        for f in glob.glob(outpath+'/*.png'):
            os.remove(f)

if __name__ == "__main__":
    main()