import numpy as np
import astropy.units as u

from sst1mpipe.io.sst1m_event_source import SST1MEventSource

MON_EVT_TYPE = 8
class sliding_pedestals:
    def __init__(self,max_array_size = 50):
        self.timestamps = np.array([])
        self.ped_array  = np.array([])
        self.max_array_size = max_array_size
    
    def add_ped_evt(self,evt):
        tel = evt.sst1m.r0.tels_with_data[0]
        pedestal  = evt.sst1m.r0.tel[tel].adc_samples
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
        return self.ped_array.mean(axis=(0,2))

    def get_charge_median(self):
        return np.median(self.ped_array.mean(axis=2),axis=0)

    def get_charge_std(self):
        return self.ped_array.std(axis=2).mean(axis=0)
    
    def fill_mon_container(self,evt):
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

    def load_firsts_pedestals(self,file_path,max_n_ped=10,max_evt=200):

        data_stream = SST1MEventSource([file_path],
                                       max_events=max_evt)
        for ii,event in enumerate(data_stream):
            tel = event.sst1m.r0.tels_with_data[0]
            r0data = event.sst1m.r0.tel[tel]
            if r0data._camera_event_type.value == MON_EVT_TYPE:
                self.add_ped_evt(event)
            if self.timestamps.shape[0] >= max_n_ped:
                break

        
            

            
