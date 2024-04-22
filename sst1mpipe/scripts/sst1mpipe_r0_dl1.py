#!/usr/bin/env python

"""
A script to calibrate raw data (R0) or MC (R1) in DL1.
- Inputs are a single raw .fits.fz data file (containing single telescope data) 
or .simtel.gz output file of sim_telarray (may contain more telescopes).
- Output is hdf file with a table of DL1 parameters.

Usage:

$> python sst1mpipe_r0_dl1.py
--input-file SST1M1_20240304_0012.fits.fz
--output-dir ./
--config sst1mpipe_config.json
--pointing-ra 85.0
--pointing-dec 25.0
--force-pointing
--px-charges
--reclean

"""

import sst1mpipe
from sst1mpipe.utils.NSB_tools import VAR_to_Idrop, get_optical_eff_shift, VAR_to_NSB
from sst1mpipe.utils import (
    correct_true_image, 
    energy_min_cut,
    remove_bad_pixels,
    add_pointing_to_events,
    add_event_id,
    add_trigger_time,
    get_tel_string,
    get_location,
    get_subarray,
    image_cleaner_setup,
    sliding_pedestals,
    swap_modules_59_88
)
from sst1mpipe.io import (
    write_extra_parameters,
    load_config,
    check_outdir,
    write_r1_dl1_cfg,
    write_assumed_pointing,
    write_pixel_charges_table,
    write_charge_images,
    read_charge_images,
    get_target,
    write_wr_timestamps,
    write_charge_fraction,
    write_dl1_info
)

from sst1mpipe.calib import (
    get_calibration_parameters,
    get_dc_to_pe,
    window_transmittance_correction,
    get_window_corr_factors,
    saturated_charge_correction
)

from ctapipe.io import EventSource
from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from ctapipe.reco import ShowerProcessor
from ctapipe.io import DataWriter
from ctapipe.instrument import SubarrayDescription

from sst1mpipe.io.sst1m_event_source import SST1MEventSource

import os
import sys
import argparse
import numpy as np
import logging

from os import path

from ctapipe.io import (
    TableLoader, 
    read_table
)

import astropy.units as u
from astropy.time import Time


def parse_args():

    parser = argparse.ArgumentParser(description="MC R1 or data R0 to DL1")

    # Required arguments
    parser.add_argument(
                    '--input-file', '-f', type=str,
                    dest='input_file',
                    help='Path to the simtelarray or data file',
                    required=True
                    )
    parser.add_argument('--config', '-c', action='store', type=str,
                    dest='config_file',
                    help='Path to a configuration file.',
                    required=True
                    )

    # Optional arguments
    parser.add_argument(
                    '--output-dir', '-o', type=str,
                    dest='outdir',
                    help='Path to store the output DL1 file',
                    default='./'
                    )

    parser.add_argument(
                    '--px-charges',
                    action='store_true',
                    help='Extract pixel charges for MC-data tuning and store their distribution in extra h5 file.',
                    dest='pixel_charges'
                    )

    parser.add_argument(
                    '--precise-timestamps',
                    action='store_true',
                    help='Store WR timestamps in the output DL1 table. Needs some extra processing time to go through the event source again.',
                    dest='precise_timestamps'
                    )

    parser.add_argument(
                    '--pointing-ra', '-r', type=float,
                    dest='ra',
                    help='Pointing RA (deg)',
                    default=None
                    )

    parser.add_argument(
                    '--pointing-dec', '-d', type=float,
                    dest='dec',
                    help='Pointing DEC (deg)',
                    default=None
                    )

    parser.add_argument(
                    '--force-pointing',
                    action='store_true',
                    help='Use pointing coordinates provided manualy by user even if there is a pointing info in the fits file.',
                    dest='force_pointing'
                    )

    parser.add_argument(
                    '--reclean',
                    action='store_true',
                    help='Perform cleaning based on pre-calculated charge distributions from pedestal events.',
                    dest='reclean'
                    )

    args = parser.parse_args()
    return args

    
def main():

    args = parse_args()

    input_file = args.input_file
    outdir = args.outdir
    ra = args.ra
    dec = args.dec
    pixel_charges = args.pixel_charges
    force_pointing = args.force_pointing
    reclean = args.reclean
    precise_timestamps = args.precise_timestamps
   
    if "simtel" in input_file:
        ismc = True
        output_file = os.path.join(outdir, input_file.split('/')[-1].rstrip(".corsika.gz.simtel.gz") + "_dl1.h5")
        output_logfile = os.path.join(outdir, input_file.split('/')[-1].rstrip(".corsika.gz.simtel.gz") + "_r1_dl1.log")
        output_file_px_charges = os.path.join(outdir, input_file.split('/')[-1].rstrip(".corsika.gz.simtel.gz") + "_pedestal_hist.h5")
    else:
        ismc = False
        output_file = os.path.join(outdir, input_file.split('/')[-1].rstrip(".fits.fz") + "_dl1.h5")
        output_logfile = os.path.join(outdir, input_file.split('/')[-1].rstrip(".fits.fz") + "_r1_dl1.log")
        output_file_px_charges = os.path.join(outdir, input_file.split('/')[-1].rstrip(".fits.fz") + "_pedestal_hist.h5")

    check_outdir(outdir)

    if reclean:
        output_logfile = os.path.join(outdir, output_logfile.split('/')[-1].rstrip(".log") + "_recleaned.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers= [
            logging.FileHandler(output_logfile, 'w+'),
            logging.StreamHandler(stream=sys.stdout)
            ]
    )

    logging.info('sst1mpipe version: %s', sst1mpipe.__version__)
    logging.info('Input file: %s', input_file)
    logging.info('Output file: %s', output_file)

    max_events = None

    config = load_config(args.config_file, ismc=ismc)

    if ismc:
        source = EventSource(input_file, max_events=max_events, allowed_tels=config["allowed_tels"])

        logging.info("Tel 1 Intensity correction factor: {}".format(config['NsbCalibrator']['intensity_correction']['tel_001']))
        logging.info("Tel 2 Intensity correction factor: {}".format(config['NsbCalibrator']['intensity_correction']['tel_002']))

    else:
        source = SST1MEventSource([input_file], max_events=max_events)
        source._subarray = get_subarray()

        ## init pedestal_info and loading first pedestal events in pedestal_info
        pedestal_info = sliding_pedestals()
        pedestal_info.load_firsts_pedestals(input_file)
        if pedestal_info.get_n_events() == 0:
            logging.warning("No pedestal events found in firsts events")
            pedestals_in_file = False
        else:
            logging.info("{} pedestals events loaded in buffer".format(pedestal_info.get_n_events()))
            pedestals_in_file = True

        logging.info("Tel 1 Intensity correction factor: {}".format(config['NsbCalibrator']['intensity_correction']['tel_021']))
        logging.info("Tel 2 Intensity correction factor: {}".format(config['NsbCalibrator']['intensity_correction']['tel_022']))

        if config['NsbCalibrator']['apply_pixelwise_Vdrop_correction']:
            logging.info(" Voltage drop correction is applyed pixelwise")

        if config['NsbCalibrator']['apply_global_Vdrop_correction']:
            logging.info(" Voltage drop correction is applyed globaly")

        if config['NsbCalibrator']['apply_global_Vdrop_correction'] == config['NsbCalibrator']['apply_pixelwise_Vdrop_correction']:
            if config['NsbCalibrator']['apply_global_Vdrop_correction']:
                logging.error(" Voltage drop correction is applyed 2 times!!! this is WRONG!")
            else:
                logging.warning("NO Voltage drop correction is applyed")

        # Reading target name and assumed pointing ra,dec from the target field
        # of the Events fits header
        target, ra_fits, dec_fits, wobble_fits = get_target(input_file)
        if (ra_fits is not None) & (dec_fits is not None):
            logging.info('Pointing info from the fits file: TARGET: ' + target + ', COORDS: ' + str(ra_fits) + ' ' + str(dec_fits) + ', WOBBLE: ' + wobble_fits)
            if force_pointing & (ra is not None) & (dec is not None):
                logging.info('Using pointing info from manual input anyway (forced by user).')
                pointing_manual=True
            else:
                ra = ra_fits
                dec = dec_fits
                output_file = output_file.split("_dl1.h5")[0] + "_" + wobble_fits + "_dl1.h5"
                output_file_px_charges = output_file_px_charges.split("_pedestal_hist.h5")[0] + "_" + wobble_fits + "_pedestal_hist.h5"
                pointing_manual=False
        elif force_pointing & (ra is not None) & (dec is not None):
            # Note that if there is no TARGET field in the fits file, it most probably means 
            # that it is a file where the shifters were tuning the trigger threshold on actual nsb conditions.
            logging.warning('No coordinates in the FITS header. Using pointing info from manual input. Are you sure that this is what you want?')
            pointing_manual=True
        else:
            logging.warning('No coordinates provided, exiting...')
            exit()

        if source.swat_event_ids_available:
            logging.info('Using arrayEvtNum as event_id: input file contains SWAT array event IDs')
        else:
            logging.info('Using eventNumber as event_id: input file does not contain SWAT array event IDs')

    if reclean:
        output_file = os.path.join(outdir, output_file.split('/')[-1].rstrip(".h5") + "_recleaned.h5")
        input_file_px_charges = output_file_px_charges
        output_file_px_charges = os.path.join(outdir, output_file_px_charges.split('/')[-1].rstrip(".h5") + "_recleaned.h5")

    r1_dl1_calibrator = CameraCalibrator(subarray=source.subarray, config=config)
    image_processor   = image_cleaner_setup(subarray=source.subarray, config=config, ismc=ismc)

    cleaner = config['ImageProcessor']['image_cleaner_type']
    if (cleaner == 'ImageCleanerSST') and not ismc:
        #to be sure about the order, sort telescope_defaults according to increasing nsb level
        for key, value in config['ImageProcessor'][cleaner]['telescope_defaults'].items():
            config['ImageProcessor'][cleaner]['telescope_defaults'][key] = sorted(value, key=lambda x: x['min_nsb_level'], reverse=True)

    if reclean:
        dl1_charges = read_charge_images(input_file_px_charges)
        dl1_charges = dl1_charges[dl1_charges['n'] > config['ImageProcessor'][cleaner]['min_number_pedestals']]
        dl1_charges['mean_charge'] = np.average(dl1_charges['average_q'], axis=1)
        ped_mean_charge = np.ndarray(shape=[0,3])

    shower_processor  = ShowerProcessor(subarray=source.subarray, config=config)

    n_pedestals = 0
    n_saturated = 0
    n_pedestals_survived = 0
    n_triggered_tel1 = 0
    n_triggered_tel2 = 0
    frac_rised = 0

    if pixel_charges:
        BINS = 1000
        N_events= 0
        N_events_tel1= 0
        N_events_tel2= 0
        if not source.is_simulation:
            final_histogram = np.zeros(BINS)
            #NOTE: To store images of stdevs. and average charges for pedestal events
            ped_q_map = []
            ped_q_sum = 0
            ped_q2_sum = 0
            ped_n = 0
            ped_time_start = None
            ped_time_window = config["analysis"]["ped_time_window"]*u.s
        else:
            final_histogram_tel1 = np.zeros(BINS)
            final_histogram_tel2 = np.zeros(BINS)

    survived_charge_fraction_1 = []
    survived_charge_fraction_2 = []

    with DataWriter(
        source, output_path=output_file, 
        overwrite        = True, 
        write_showers    = True,
        write_parameters = True,
        write_images     = True,

    ) as writer:
        
        for i, event in enumerate(source):

            if not source.is_simulation:

                # NOTE: This needs to be changed in the future when event source hopefuly provides events with both telescope data
                if i == 0:
                    tel = event.sst1m.r0.tels_with_data[0]
                    calibration_parameters, calib_file = get_calibration_parameters(telescope=tel, config=config)
                    dc_to_pe = get_dc_to_pe(calibration_parameters)
                    window_corr_factors, window_file = get_window_corr_factors(telescope=tel, config=config)
                    tel_string = get_tel_string(tel, mc=False)
                    location = get_location(config=config, tel=tel_string)

                    if config['swap_modules_59_88'][tel_string]:
                        logging.info('Swapping wrongly connected modules 59 and 88 for ' + tel_string)

                event.trigger.tels_with_trigger = [tel]
                r0data = event.sst1m.r0.tel[tel]

                baseline_subtracted = (r0data.adc_samples.T - r0data.digicam_baseline)

                ## Apply (or not) pixel wise Voltage drop correction
                ## TODO ?? TOTEST
                if config['NsbCalibrator']['apply_pixelwise_Vdrop_correction']:
                    VI = VAR_to_Idrop(pedestal_info.get_charge_std()**2, tel)
                else:
                    VI = 1.0
                event.r1.tel[tel].waveform = (baseline_subtracted / dc_to_pe /VI ).T
                event.r1.tel[tel].selected_gain_channel = np.zeros(source.subarray.tel[tel].camera.readout.n_pixels,dtype='int8')

                event_type = event.sst1m.r0.tel[tel]._camera_event_type.value

                # Fill trigger container properly
                # SHOULD BE REMOVED as soon as event source can handle this
                event = add_trigger_time(event, telescope=tel)

                # Add assumed pointing (this should be part of Event Source in the future)
                # This stores the pointing information in the right containters. If done this way, pointing information is automaticaly propagated in
                # the output DL1 file, in /dl1/monitoring/subarray/pointing and /dl1/monitoring/telescope/pointing/TEL
                event = add_pointing_to_events(event, ra=ra, dec=dec, telescope=tel, location=location)

                # Adding event_id and obs_id in event.index
                # event_id is in event.sst1m.r0.event_id, but obs_id must be made up
                # SHOULD BE REMOVED as soon as event source can handle this
                event = add_event_id(event, filename=output_file, event_number=i)

                # Here we swap two wrongly connected modules in tel2 after 
                # camera refurbishment in 2023. Only R1 waveforms are swapped.
                if config['swap_modules_59_88'][tel_string]:
                    event = swap_modules_59_88(event, tel=tel)

                ## Fill monitoring container with baseline info :
                if event_type==8:
                    pedestal_info.add_ped_evt(event)
                    pedestal_info.fill_mon_container(event)
                    

            # For an unknown reason, event.simulation.tel[tel].true_image is sometime None, which kills the rest of the script
            # and simulation histogram is not saved. Here we repace it with an array of zeros.
            if source.is_simulation:
                event = correct_true_image(event)

            # This function flags the bad pixel according to the cfg file, and just for sure also kills the waveforms.
            # Charges in these pixels are then interpolated using method set in cfg: invalid_pixel_handler_type
            # Default is NeighborAverage, but can be turned off with 'null'
            event = remove_bad_pixels(event, config=config)

            #set proper charge info according to time bins of pedestal events
            if reclean and (len(dl1_charges) > 0):
                for [start_time, n, Qped, sig_Qped, meanQ] in reversed(dl1_charges):
                    if event.trigger.time >= start_time:
                        break
                image_processor.clean.average_charge = Qped
                image_processor.clean.stdev_charge = sig_Qped
                image_processor.clean.nsb_level = meanQ
                image_processor.clean.config = config['ImageProcessor'][cleaner]
                ped_mean_charge = np.append(ped_mean_charge, [[event.index.obs_id, event.index.event_id, meanQ]], axis=0)

            r1_dl1_calibrator(event) # r1->dl1a (images, peak times)

            if reclean and (len(dl1_charges) > 0):
                frac_rised += image_processor.clean.frac_rised

            if not source.is_simulation:

                # Integration correction of saturated pixels
                event, saturated = saturated_charge_correction(event, 
                adc_samples=baseline_subtracted,
                telescope=tel
                )
                if saturated: n_saturated += 1

                event = window_transmittance_correction(
                    event, 
                    window_corr_factors=window_corr_factors, 
                    telescope=tel
                    )

            image_processor(event) # dl1a->dl1b (hillas parameters)

            # Extraction of pixel charge distribution for MC-data tuning
            if pixel_charges:
                if not source.is_simulation:
                    event_type = event.sst1m.r0.tel[tel]._camera_event_type.value
                    if ped_time_start is None:
                        ped_time_start = event.trigger.time
                    if event_type == 8:
                        image = event.dl1.tel[tel].image
                        hist, bin_edges = np.histogram(image, range=(-10, 40), bins=BINS, density=False)
                        final_histogram += hist
                        N_events += 1
                        time_diff = event.trigger.time - ped_time_start
                        if (time_diff > ped_time_window) and (ped_n > 0):
                            mean = ped_q_sum/ped_n
                            stdev = np.sqrt(ped_q2_sum/ped_n-mean*mean)
                            ped_q_map.append([ped_time_start, ped_n, mean, stdev])
                            ped_n = 0
                            ped_q_sum = 0
                            ped_q2_sum = 0
                            ped_time_start = event.trigger.time
                        ped_q_sum += image
                        ped_q2_sum += image*image
                        ped_n += 1

                else:
                    for tel in event.trigger.tels_with_trigger:
                        # We need to get rid of shower pixels
                        noise_mask = ~np.array(event.simulation.tel[tel].true_image, dtype=bool)
                        image = event.dl1.tel[tel].image[noise_mask]
                        hist, bin_edges = np.histogram(image, range=(-10, 40), bins=BINS, density=False)
                        if tel == 1:
                            final_histogram_tel1 += hist
                            N_events_tel1 += 1
                        elif tel == 2:
                            final_histogram_tel2 += hist
                            N_events_tel2 += 1

            # We would like to store in DL1 also some additional parameters needed for disp reconstruction and few more additional features
            # It cannot be done at this level, because: AttributeError: 'CameraHillasParametersContainer' object has no attribute 'disp'

            shower_processor(event) # dl1b->dl2 (reconstruction of stereo parameters, also energy/direction/classification in the future versions of ctapipe)
            
            # Counting all triggered events
            if source.is_simulation:
                tels = event.trigger.tels_with_trigger
                for tel in tels:
                    if tel == 1:
                        n_triggered_tel1 += 1
                    elif tel == 2:
                        n_triggered_tel2 += 1
            else:
                if tel == 21:
                    n_triggered_tel1 += 1
                elif tel == 22:
                    n_triggered_tel2 += 1

            # Counting pedestal events in the file and skipping them for the output file
            if not source.is_simulation:
                if event_type == 8:

                    n_pedestals += 1
                    # writing pedestal info in dl1
                    if (n_pedestals%21==20):
                        writer._writer.write(
                            table_name='dl1/monitoring/telescope/pedestal',
                            containers=[event.mon.tel[tel].pedestal],
                        )
                    if np.isfinite(event.dl1.tel[tel].parameters.hillas.intensity) :
                        n_pedestals_survived += 1
                    continue

            # Calculation of fraction of true charge which survived cleaning
            if source.is_simulation:
                tels = event.trigger.tels_with_trigger
                for tel in tels:
                    cleaning_mask = event.dl1.tel[tel].image_mask
                    if tel == 1:
                        survived_charge_fraction_1.append(sum(event.simulation.tel[tel].true_image[cleaning_mask])/sum(event.simulation.tel[tel].true_image))
                    elif tel == 2:
                        survived_charge_fraction_2.append(sum(event.simulation.tel[tel].true_image[cleaning_mask])/sum(event.simulation.tel[tel].true_image))
                    else:
                        logging.warning('Telescope %f not recognized, survived charge fraction not logged.', tel)

            ## Correct (or not) the Voltage drop effect : Global correction on the intensity
            ## apply (or not) some absolute correction on the intensity
            
            if not source.is_simulation:
                I0 = event.dl1.tel[tel].parameters.hillas.intensity
                if config['NsbCalibrator']['apply_global_Vdrop_correction']:
                    VI = VAR_to_Idrop (np.median(pedestal_info.get_charge_std()**2),
                                       tel)
                    I_corr = I0/VI*config['NsbCalibrator']["intensity_correction"][tel_string]
                else:
                    I_corr = I0*config['NsbCalibrator']["intensity_correction"][tel_string]
                event.dl1.tel[tel].parameters.hillas.intensity = I_corr
            else:
                for tel in event.trigger.tels_with_trigger:
                    tel_string = get_tel_string(tel, mc=True)
                    I0 = event.dl1.tel[tel].parameters.hillas.intensity
                    I_corr = I0*config['NsbCalibrator']["intensity_correction"][tel_string]
                    event.dl1.tel[tel].parameters.hillas.intensity = I_corr

            writer(event)

        if max_events == None and source.is_simulation:
            writer.write_simulation_histograms(source)

    # Write additional params in the DL1 file
    # - these are not defined in the ctapipe containers, but are necessary for (mono) reconstruction
    # - DISP parameters are calculated and stored
    # - some more parameters are extracted from other tables in the file and added to the parameters table for convenience
    # NOTE: unfortunately using this, units in all columns have to be dropped, because otherwise ctapipe merging tool fails.
    # I didn't find a solution, this should definitely be revisited!
    if reclean and (len(dl1_charges) > 0):
        write_extra_parameters(output_file, config=config, ismc=ismc, meanQ=ped_mean_charge)
    else:
        write_extra_parameters(output_file, config=config, ismc=ismc)

    if source.is_simulation:
        write_charge_fraction(
            output_file, 
            survived_charge={
                "tel_001": survived_charge_fraction_1, 
                "tel_002": survived_charge_fraction_2
                }
            )

    # Write WR timestamps with high numerical precision
    if not source.is_simulation and precise_timestamps:
        write_wr_timestamps(output_file, event_source=SST1MEventSource([input_file], max_events=max_events))

    # Write pointing information in the main DL1 table for convenience
    if not source.is_simulation:
        write_assumed_pointing(output_file, config=config, pointing_ra=ra, pointing_dec=dec)

    logging.info('Total number of TEL1 triggered events in the file: %d', n_triggered_tel1)
    logging.info('Total number of TEL2 triggered events in the file: %d', n_triggered_tel2)

    if not source.is_simulation:
        logging.info('Total number of saturated events in the file: %d', n_saturated)
        logging.info('Total number of pedestal events in the file: %d', n_pedestals)
        if n_pedestals > 0:
            logging.info('Fraction of pedestal events that survived cleaning: %f', n_pedestals_survived/n_pedestals)
        else:
            logging.info('No pedestal events found!')

    if reclean and (len(dl1_charges) > 0):
            logging.info('Average (per event) fraction of pixels (N/1296) with raised picture threshold: %f', frac_rised/i)

    if source.is_simulation:
        # Cut on minimum mc_energy in the output file, which is needed if we want to safely combine MC from different productions
        # NOTE: This doesn't change the mc and histogram tab in the output files and this must be taken care of in performance
        # evaluation. We cannot recalculate N of simulated events at this point for each individual dl1 file, because it would 
        # lead to an error of the order of 10%.
        energy_min_cut(output_file, config=config)

    if source.is_simulation:
        write_dl1_info(
            output_file,
            n_triggered_tel1=n_triggered_tel1,
            n_triggered_tel2=n_triggered_tel2
        )
    else:
        write_dl1_info(
            output_file, 
            target=target,
            ra=ra,
            dec=dec,
            manual_coords=pointing_manual,
            wobble=wobble_fits,
            calib_file=calib_file, 
            window_file=window_file, 
            n_saturated=n_saturated, 
            n_pedestal=n_pedestals, 
            n_survived_pedestals=n_pedestals_survived,
            n_triggered_tel1=n_triggered_tel1,
            n_triggered_tel2=n_triggered_tel2,
            swat_event_ids_used=source.swat_event_ids_available
            )

    # We write calibration configuration in the output file
    # NOTE: If one use the ctapipe merging tool this table is missing in the merged DL1 file!
    # TODO: Broken after implementation telescope dependent tailcuts, but not supper important
    # write_r1_dl1_cfg(output_file, config=config)

    # Save pixel charges histograms and maps in output file
    if pixel_charges:

        if not source.is_simulation:

            if ped_n > 0:
                mean = ped_q_sum/ped_n
                stdev = np.sqrt(ped_q2_sum/ped_n-mean*mean)
                ped_q_map.append([ped_time_start, ped_n, mean, stdev])

                if (N_events > 0):
                    data = np.array(final_histogram)[..., np.newaxis]
                    names = ['pixel_charge']
                    write_charge_images(ped_q_map, output_file=output_file_px_charges)
            else:
                logging.warning('There are no pedestal events in the file to calculate pixel charges distributions.')
        else:
            if ((N_events_tel1 > 0) and (N_events_tel2 > 0)):
                data = np.column_stack((np.array(final_histogram_tel1), np.array(final_histogram_tel2)))
                names = ['pixel_charge_tel1', 'pixel_charge_tel2']
        
        if (N_events > 0) or ((N_events_tel1 > 0) and (N_events_tel2 > 0)):
            write_pixel_charges_table(data, bin_edges, names=names, output_file=output_file_px_charges)
        else:
            logging.warning('There are no pedestal events in the file to fill the pixels charge histogram.')


if __name__ == '__main__':
    main()
