import numpy as np
import pkg_resources
import pandas as pd
from os import path
import logging
from sst1mpipe.utils import get_tel_string

def get_dc_to_pe(calibration_parameters):
    """
    Gets ADC -> p.e. conversion factor from the 
    calibration file. Gain drop is not taken 
    int account!

    Parameters
    ----------
    calibration_parameters: pandas.DataFrame

    Returns
    -------
    dc_to_pe: numpy.ndarray

    """

    dc_to_pe = np.array(calibration_parameters['dc_to_pe']) 
    ok_mask = calibration_parameters['calib_flag']==1
    dc_to_pe[~ok_mask] = dc_to_pe[ok_mask].mean()
    return dc_to_pe


def get_default_calibration(telescope=None):
    """
    Provides default calibration file, used in 
    the case when it is not defined in the 
    configuration file.

    Parameters
    ----------
    telescope: int
        Telescope number as in
        event.sst1m.r0.tels_with_data

    Returns
    -------
    calibration_parameters: pandas.DataFrame
    calib_file: string

    """

    if (telescope == 21) or (telescope == 1):
        default_calib_file_tel1 = '20230917_0528_0529_Tel1_fitted_parameters.h5'
        logging.info('Calib file used: ' + default_calib_file_tel1)
        calibration_file = pkg_resources.resource_filename('sst1mpipe',path.join('data',default_calib_file_tel1))
    elif (telescope == 22) or (telescope == 2):
        default_calib_file_tel2 = '20230921_0149_0150_Tel2_fitted_parameters.h5'
        logging.info('Calib file used: ' + default_calib_file_tel2)
        calibration_file = pkg_resources.resource_filename('sst1mpipe',path.join('data', default_calib_file_tel2)) 
    else:
        logging.error('Telescope {} not known'.format(tel))
    calibration_parameters = pd.read_hdf(calibration_file)
    return calibration_parameters, calibration_file


def get_calibration_parameters(telescope=None, config=None):
    """
    Finds and reads the calibration file.

    Parameters
    ----------
    telescope: int
        Telescope number as in
        event.sst1m.r0.tels_with_data
    config: dict

    Returns
    -------
    calibration_parameters: pandas.DataFrame
    calib_file: string

    """

    if "telescope_calibration" in config:
        if config["telescope_calibration"]["tel_" + str(telescope).zfill(3)]:
            calib_file = config["telescope_calibration"]["tel_" + str(telescope).zfill(3)]
            calibration_parameters = pd.read_hdf(calib_file)
            logging.info("Calibration File for Tel %s: %s", telescope, calib_file)
        else:
            logging.info("NO CALIBRATION FILE FOR TELESCOPE %s FOUND IN THE CFG FILE, DEFAULT CALIBRATION USED.", telescope)
            calibration_parameters, calib_file = get_default_calibration(telescope=telescope)
    else:
        logging.info("NO CALIBRATION FILE FOR TELESCOPE %s FOUND IN THE CFG FILE, DEFAULT CALIBRATION USED.", telescope)
        calibration_parameters, calib_file = get_default_calibration(telescope=telescope)
    return calibration_parameters, calib_file


def get_default_window(telescope=None):
    """
    Provides default window transmissivity file, 
    used in the case when it is not defined in the 
    configuration file.

    Parameters
    ----------
    telescope: int
        Telescope number as in
        event.sst1m.r0.tels_with_data

    Returns
    -------
    window_corr: numpy.ndarray
    window_file: string

    """

    if (telescope == 21) or (telescope == 1):
        default_window_file_tel1 = 'corr_factor_1st_wdw.txt'
        logging.info('Window file used: ' + default_window_file_tel1)
        window_file = pkg_resources.resource_filename('sst1mpipe',path.join('data', default_window_file_tel1))
    elif (telescope == 22) or (telescope == 2):
        default_window_file_tel2 = 'corr_factor_2nd_wdw.txt'
        logging.info('Window file used: ' + default_window_file_tel2)
        window_file = pkg_resources.resource_filename('sst1mpipe',path.join('data', default_window_file_tel2)) 
    else:
        logging.error('Telescope {} not known'.format(tel))
    window_corr = np.loadtxt(window_file, unpack=True, skiprows=1, usecols=1)
    return window_corr, window_file


def get_window_corr_factors(telescope=None, config=None):
    """
    Finds and reads the window transmissivity file.

    Parameters
    ----------
    telescope: int
        Telescope number as in
        event.sst1m.r0.tels_with_data
    config: dict

    Returns
    -------
    window_corr: numpy.ndarray
    window_file: string

    """

    if "window_transmittance" in config:
        if config["window_transmittance"]["tel_" + str(telescope).zfill(3)]:
            window_file = config["window_transmittance"]["tel_" + str(telescope).zfill(3)]
            window_corr = np.loadtxt(window_file, unpack=True, skiprows=1, usecols=1)
            logging.info("Window File for Tel %s: %s", telescope, window_file)
        else:
            logging.info("NO WINDOW TRANSMITTANCE FILE FOR TELESCOPE %s FOUND IN THE CFG FILE, DEFAULT WINDOW USED.", telescope)
            window_corr, window_file = get_default_window(telescope=telescope)
    else:
        logging.info("NO WINDOW TRANSMITTANCE FILE FOR TELESCOPE %s FOUND IN THE CFG FILE, DEFAULT WINDOW USED.", telescope)
        window_corr, window_file = get_default_window(telescope=telescope)

    # Correct for wrongly mapped pixels
    tel_string = get_tel_string(telescope, mc=False)
    if config['swap_modules_59_88'][tel_string]:
        # module 59
        mask59 = np.zeros(1296, dtype=bool)
        mask59[1029] = True
        mask59[1098:1102+1] = True
        mask59[1133:1134+1] = True
        mask59[1064:1067+1] = True
        window_corr_59 = window_corr[mask59]
        
        # module 88
        mask88 = np.zeros(1296, dtype=bool)
        mask88[1103] = True
        mask88[1165:1169+1] = True
        mask88[1194:1195+1] = True
        mask88[1135:1138+1] = True
        window_corr_88 = window_corr[mask88]
        
        window_corr[mask59] = window_corr_88
        window_corr[mask88] = window_corr_59  

    return window_corr, window_file


def window_transmittance_correction(
        event, window_corr_factors=None, 
        telescope=None):
    """
    Applies window transmittance correction 
    on the integrated waveforms (images)

    Parameters
    ----------
    event: 
        sst1mpipe.io.containers.SST1MArrayEventContainer
    window_corr_factors: numpy.ndarray
    telescope: int
        Telescope number as in
        event.sst1m.r0.tels_with_data

    Returns
    -------
    event:
        sst1mpipe.io.containers.SST1MArrayEventContainer

    """

    image_corrected = event.dl1.tel[telescope].image / window_corr_factors
    event.dl1.tel[telescope].image = image_corrected.astype(np.float32) 

    return event


def saturated_charge_correction(event, adc_samples=None, telescope=None):
    """
    Finds saturated waveforms and applies different peak integration on
    them, as the standard one does not perform well in such cases. This
    method integrates the peak above 20\% of the amplitude.
    Peak time for saturated events is also corrected as the middle of 
    the integration window.

    Parameters
    ----------
    event: 
        sst1mpipe.io.containers.SST1MArrayEventContainer
    adc_samples: numpy.ndarray
        Baseline subtracted waveforms
    telescope: int
        Telescope number as in
        event.sst1m.r0.tels_with_data

    Returns
    -------
    event:
        sst1mpipe.io.containers.SST1MArrayEventContainer
    saturated: bool
        True if integration correction was applied on at least 
        one pixel waveform
    """

    saturated_threshold = 3000
    width_level = 2500
    width_threshold = 5
    integration_level = 0.2

    # saturated pixels
    mask_saturated = np.max(adc_samples, axis=0) > saturated_threshold
    saturated = False

    if sum(mask_saturated) > 0:

        image_new = event.dl1.tel[telescope].image
        peaktime_new = event.dl1.tel[telescope].peak_time

        # iterate over baseline subtracted waveforms and correct integration of those peaking above
        # saturation threshold and with larger width
        for k, w in enumerate(adc_samples.T):
            
            if mask_saturated[k]:
                mask_width = w > width_level

                n_samples = mask_width.shape[0]

                max_adc = max(w)
                mask_integration = w > integration_level * max_adc
                integration_start = np.arange(0, n_samples)[w >= integration_level * max_adc][0]

                # This is needed to avoid secondary peaks (it looks for first drop below 20 percent after maximum)
                # We also select the first one, if there is a plato in the small peak in the wavefrom, which sometimes happen
                index_max = np.arange(0, n_samples)[w == max_adc][0]
                int_stop = np.arange(0, n_samples)[(w < integration_level * max_adc) & (np.arange(0, n_samples) > index_max)]
                # If it does not find where to stop it means the waveform extends over the readout window
                if len(int_stop) > 0:
                    integration_stop = int_stop[0]
                else:
                    integration_stop = n_samples-1

                width = sum(mask_width)
                peak_sample = width/2 + np.arange(0, n_samples)[mask_width][0]
                peak_time = peak_sample * 4

                if width > width_threshold:

                    # Peak integration correction
                    image_new[k] = sum(event.r1.tel[telescope].waveform[k][integration_start:integration_stop+1])
                    peaktime_new[k] = peak_time
                    saturated = True

        if saturated:
            event.dl1.tel[telescope].image = image_new
            event.dl1.tel[telescope].peak_time = peaktime_new

    return event, saturated