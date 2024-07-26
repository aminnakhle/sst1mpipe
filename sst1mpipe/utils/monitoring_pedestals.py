import numpy as np
import astropy.units as u

from sst1mpipe.io.sst1m_event_source import SST1MEventSource

from sst1mpipe.calib import Calibrator_R0_R1
from ctapipe.calib import CameraCalibrator
from ctapipe.image import ImageProcessor
from sst1mpipe.utils import get_subarray

import logging

MON_EVT_TYPE = 8
class sliding_pedestals:
    def __init__(self, input_file=None, max_array_size = 300, config=None):

        self.timestamps = np.array([])
        self.ped_array  = np.array([])
        self.max_array_size = max_array_size
        self.processed_pedestals = 0
        self.config = config
        self.input_file = input_file
        self.pedestals_in_file = True

        self.log_pedestal_settings()
        self.load_firsts_pedestals()

        if self.get_n_events() == 0:
            logging.warning("No pedestal events found in firsts events. Cleaned shower/NSB events used instead.")
            self.load_firsts_fake_pedestals()
            logging.info("{} fake pedestals events loaded in buffer".format(self.get_n_events()))
            self.pedestals_in_file = False
        else:
            logging.info("{} pedestals events loaded in buffer".format(self.get_n_events()))
            self.pedestals_in_file = True

    
    def add_ped_evt(self, evt, cleaning_mask=None):
        tel = evt.sst1m.r0.tels_with_data[0]
        pedestal  = evt.sst1m.r0.tel[tel].adc_samples
        self.processed_pedestals = self.processed_pedestals + 1
        if cleaning_mask is not None:
            pedestal[cleaning_mask] = -100 * np.ones(pedestal.shape[1])
        timestamp = evt.sst1m.r0.tel[tel].local_camera_clock/1e9
        
        if self.ped_array.shape[0] == 0:
            self.ped_array = np.array([pedestal])
            self.timestamps = np.array([timestamp])
        else:
            self.ped_array = np.vstack((self.ped_array,[pedestal]))
            self.timestamps = np.append(self.timestamps,timestamp)
        if self.ped_array.shape[0] > self.max_array_size:
            self.ped_array = self.ped_array[1:]
            self.timestamps = self.timestamps[1:]
            
    def get_n_events(self):
        return self.ped_array.shape[0]

    def get_mean_ts(self):
        return self.timestamps.mean() *u.s

    def get_min_ts(self):
        return self.timestamps[0] *u.s

    def get_max_ts(self):
        return self.timestamps[-1] *u.s

    def get_charge_mean(self):
        pedarray = self.ped_array
        masked = np.ma.masked_values(pedarray, -100)
        return masked.mean(axis=(0,2)).data

    def get_charge_median(self):
        pedarray = self.ped_array
        masked = np.ma.masked_values(pedarray, -100)
        return np.median(masked.mean(axis=2).data,axis=0)

    def get_charge_std(self):
        pedarray = self.ped_array
        masked = np.ma.masked_values(pedarray, -100)
        return masked.std(axis=2).data.mean(axis=0)
    
    def fill_mon_container(self, evt):
        tel = evt.sst1m.r0.tels_with_data[0]
        mon_container=evt.mon.tel[tel].pedestal
        mon_container.n_events        = self.get_n_events()
        mon_container.sample_time     = self.get_mean_ts()
        mon_container.sample_time_min = self.get_min_ts()
        mon_container.sample_time_max = self.get_max_ts()
        mon_container.charge_mean     = self.get_charge_mean()
        mon_container.charge_median   = self.get_charge_median()
        mon_container.charge_std      = self.get_charge_std()
        return

    def load_firsts_pedestals(self,max_n_ped=50,max_evt=200):


        data_stream = SST1MEventSource([self.input_file],
                                       max_events=max_evt)
        for ii,event in enumerate(data_stream):
            tel = event.sst1m.r0.tels_with_data[0]
            r0data = event.sst1m.r0.tel[tel]
            if r0data._camera_event_type.value == MON_EVT_TYPE:
                self.add_ped_evt(event)
            if self.timestamps.shape[0] >= max_n_ped:
                break
    
    def load_firsts_fake_pedestals(self, max_evt=10):
        
        source = SST1MEventSource([self.input_file],
                                       max_events=max_evt)
        source._subarray = get_subarray()
        r1_dl1_calibrator = CameraCalibrator(subarray=source.subarray, config=self.config)

        # Here (for the first few events) we use just the simple ImageProcessor, nothing fancy
        self.config["ImageProcessor"]["image_cleaner_type"] = "TailcutsImageCleaner"
        image_processor   = ImageProcessor(subarray=source.subarray, config=self.config)

        for ii,event in enumerate(source):

            if ii == 0:
                tel = event.sst1m.r0.tels_with_data[0]
                calibrator_r0_r1 = Calibrator_R0_R1(config=self.config, telescope=tel)

            event = calibrator_r0_r1.calibrate(event)
            event.r1.tel[tel].selected_gain_channel = np.zeros(source.subarray.tel[tel].camera.readout.n_pixels,dtype='int8')

            r1_dl1_calibrator(event)
            image_processor(event)

            clenaning_mask = event.dl1.tel[tel].image_mask
            # Arbitrary cut, just to prevent too big showers from being used 
            if sum(clenaning_mask) < 20:
                self.add_ped_evt(event, cleaning_mask=clenaning_mask)

    def log_pedestal_settings(self):

        if self.config['NsbCalibrator']['apply_pixelwise_Vdrop_correction']:
            logging.info("Voltage drop correction is applied pixelwise")

        if self.config['NsbCalibrator']['apply_global_Vdrop_correction']:
            logging.info("Voltage drop correction is applied globaly")

        if self.config['NsbCalibrator']['apply_global_Vdrop_correction'] == self.config['NsbCalibrator']['apply_pixelwise_Vdrop_correction']:
            if self.config['NsbCalibrator']['apply_global_Vdrop_correction']:
                logging.error("Voltage drop correction is applied 2 times!!! this is WRONG!")
            else:
                logging.warning("NO Voltage drop correction is applied")