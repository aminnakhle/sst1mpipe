#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created June 01 2024
"""

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import glob

import astropy.units as u
import astropy.constants as c

from astropy.coordinates import SkyCoord, AltAz
from astropy.time import Time
from astropy.io import fits
from astropy.table import Table, vstack

import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import glob


import matplotlib.dates as mdates
from datetime import datetime


def get_slow_data_table(dl3_file,file_radical='DIGICAM',root_dir='/net'):
    dl3 = fits.open(dl3_file)
    
    datestr = str(dl3[1].header["OBS_ID"])[:8]
    oyear   = str(dl3[1].header["OBS_ID"])[:4]
    omonth  = str(dl3[1].header["OBS_ID"])[4:6]
    oday    = str(dl3[1].header["OBS_ID"])[6:8]
    try:
        itel = int(dl3[1].header['TELESCOP'][-2:])%20
    except:
        print("This dont work for setereo DL3")
    
    tstart = Time(dl3[1].header['TSTART'],format='unix',scale='utc')
    tstop  = Time(dl3[1].header['TSTOP'],format='unix',scale='utc')
    
    files = glob.glob('{}/cs{}/data/aux/{}/{}/{}/SST1M_TEL{}/{}{}_{}_*.fits'.format(root_dir,
                                                                                    itel,
                                                                                    oyear,
                                                                                    omonth,
                                                                                    oday,
                                                                                    itel,
                                                                                    file_radical,
                                                                                    itel,
                                                                                    datestr))
    table = vstack([ Table(fits.open(f)[1].data) for f in files])
    t_mask = (table['TIMESTAMP']>tstart.unix*1000) & (table['TIMESTAMP']<tstop.unix*1000)
    return table[t_mask]

def get_bad_intervals(table,gapmax=2000,tpas=60000):
    sat_mask = table['err_udpktarena_nfull']<gapmax
    if (~sat_mask).sum()==0:
        return []
    tt = t0  = table['TIMESTAMP'][~sat_mask][0]
    NGTI =[]

    for ii,t in enumerate(table['TIMESTAMP'][~sat_mask]):
        if np.abs(t-tt) > tpas:
            NGTI.append([t0,tt])
            t0=t
        tt=t
    NGTI.append([t0,t])
    return NGTI



def plot_rates_from_slow_data(dl3_files):
    f,ax =plt.subplots(figsize=(15,6))

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    for dl3_file in sorted(dl3_files):
        obs_id = dl3_file.split("_")[-2]
        table = get_slow_data_table(dl3_file)
        table_dsc = get_slow_data_table(dl3_file,file_radical='DigicamSlowControl')

        p = ax.plot([Time(tt/1000.,format='unix').to_datetime() for tt in table['TIMESTAMP'] ] ,
                     table['rate_swat_arrevent_nreads'],
                     '.',
                     alpha=.3,
                     label='obs_id : {}'.format(obs_id),
                     #label='rate_swat_arrevent_nreads (rate of array events successfully read from SWAT)'
                     )


        ax.plot([Time(tt/1000.,format='unix').to_datetime() for tt in table_dsc['TIMESTAMP'] ] ,
                     table_dsc['triggerStatus'][:,3],
                     '-',
                     alpha=.3,
                     color='black',
                     )#label='rate from Digicam slow control')



        NGTI = get_bad_intervals(table)
        for interval in NGTI:
            if len(NGTI)>0:
                ax.axvspan(Time(interval[0]/1000.,format='unix').to_datetime(),
                           Time(interval[1]/1000.,format='unix').to_datetime(),
                           alpha=0.2, color='red')
    ax.set_yscale('log')
    ax.set_ylabel('Rate [Hz]')
    plt.legend()
    return ax


