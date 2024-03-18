import os
import multiprocessing as mp
import numpy as np
import glob
import copy
import sys
import json
import shutil
import logging
import datetime 

from pathlib import Path
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u

'''
 USAGE ::
this run the analysis pipeline up to stereo dl2 on Calculus.
USAGE :
nohup python daily_analysis.py /path/to/some/ana_config.json 2>daily_ana.err 1>daily_ana.out &

or 

for auto proccessing of the last night data :
python daily_analysis.py 2>daily_ana.err 1>daily_ana.out

'''


DEFAULT_CONFIG_FILE = '/data/work/analysis/Daily_analysis/default_daily_ana_config.json'

class iargs:
    def __init__(self):
        pass
        

def args_maker(arg,file_list):

    args_list = []
    
    for input_file in file_list:
        t_arg = copy.copy(arg)
        t_arg.input_file=input_file
        args_list.append(t_arg)
    return args_list

    
    

def r0_dl1_1file(arg):

        cmd = 'sst1mpipe_r0_dl1 --input-file {} \
               --config {} \
               --output-dir {} --precise-timestamps'.format(arg.input_file,
                                                            arg.config_file,
                                                            arg.out_dir)
        logging.info(cmd)
        os.system(cmd)

def dl1_dl1_1file(arg):
        cmd = 'sst1mpipe_data_dl1_dl1_stereo \
               --input-file {} \
               --config {} \
               --input-dir-tel2 {} \
               --output-dir {}'.format(arg.input_file,
                                       arg.config_file,
                                       arg.tel2_dir,
                                       arg.out_dir)
        logging.info(cmd)
        os.system(cmd)


def dl1_dl2_1file(arg):
        cmd = 'sst1mpipe_dl1_dl2 \
               --input-file {} \
               --config {} \
               --models-dir {} \
               --output-dir {} --stereo'.format(arg.input_file,
                                         arg.config_file,
                                         arg.models_dir,
                                         arg.out_dir)
        logging.info(cmd)
        os.system(cmd)


def dl1_dl2_1file_1tel(arg):
        cmd = 'sst1mpipe_dl1_dl2 \
               --input-file {} \
               --config {} \
               --models-dir {} \
               --output-dir {}'.format(arg.input_file,
                                       arg.config_file,
                                       arg.models_dir,
                                       arg.out_dir)
        logging.info(cmd)
        os.system(cmd)

def dl2_dl3_1dir(arg):
        cmd = 'sst1mpipe_data_dl2_dl3 \
               --input-dir {} \
               --irf-dir {} \
               --config {} \
               --target-name {} \
               --target-ra {} \
               --target-dec {} \
               --output-dir {}'.format(arg.input_dir,
                                       arg.irf_dir,
                                       arg.config_file,
                                       arg.target_name,
                                       arg.target_ra,
                                       arg.target_dec,
                                       arg.out_dir)
        logging.info(cmd)
        os.system(cmd)


def make_runlist(tel,year,month,day,file_range):
        runlist = []
        basedir = '/net/cs{}/data/raw/'.format(tel)
        tel_str = 'tel{}'.format(tel)
        for run in range(file_range[0],file_range[1]+1):
                datestr = "{}{:02d}{:02d}".format(year,month,day)
                filerad = 'SST1M{}'.format(tel)
                filepath = os.path.join(basedir,
                                       str(year),
                                       "{:02d}".format(month),
                                       "{:02d}".format(day),
                                       filerad,
                                       '{}_{}_{:04d}.fits.fz'.format(filerad,
                                                                     datestr,
                                                                     run)
                                       )
                runlist.append(filepath)
        return runlist

def make_runlist_allfiles(tel,year,month,day):
        runlist = []
        basedir = '/net/cs{}/data/raw/'.format(tel)
        tel_str = 'tel{}'.format(tel)
        datestr = "{}{:02d}{:02d}".format(year,month,day)
        filerad = 'SST1M{}'.format(tel)
        filedir = os.path.join(basedir,
                                       str(year),
                                       "{:02d}".format(month),
                                       "{:02d}".format(day),
                                       filerad
                                       )
        return glob.glob(filedir+"/"+filerad+"*.fits.fz")

def refine_file_list(file_list):
    list_dict = dict()
    list_dict['DARK']       = []
    list_dict['Transition'] = []
    list_dict['UNKNOWN']    = []
    for fitsfile in file_list:
        f = fits.open(fitsfile)
        try:
            target=f[2].header['TARGET'].split(',')[0].split('_')[0].replace(" ", "")
        except:
            target='UNKNOWN'
        if target not in list_dict.keys():
            list_dict[target]    = []
        list_dict[target].append(fitsfile)
        f.close()
    return list_dict

def run_daily_ana(daily_config_file):

    with open(daily_config_file) as json_data_file:
        daily_config = json.load(json_data_file)

    n_proc           = daily_config["n_proc"]
    ana_dir          = daily_config["ana_dir"]
    config_file      = daily_config["config_file"]
    year             = daily_config["year"]
    month            = daily_config["month"]
    day              = daily_config["day"]
    mono_model_dir   = daily_config["mono_model_dir"]
    stereo_model_dir = daily_config["stereo_model_dir"]
    cs1              = daily_config["cs1"]
    cs2              = daily_config["cs2"]
    stereo           = daily_config["stereo"]


    if (year is None) or (month is None) or (day is None):
        logging.info("Running daily analysis for last night !")
        year  = (datetime.datetime.now()-datetime.timedelta(days=1)).year
        month = (datetime.datetime.now()-datetime.timedelta(days=1)).month
        day   = (datetime.datetime.now()-datetime.timedelta(days=1)).day


    try:
        output_logfile = os.path.join(ana_dir, "logs", 'daily_ana_{:04d}{:02d}{:02d}.log'.format(year,month,day))
    except:
        output_logfile = os.path.join(ana_dir, "logs", 'default_daily_ana.log')
    
    Path(ana_dir + '/logs/').mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers= [
            logging.FileHandler(output_logfile, 'w+'),
            logging.StreamHandler(stream=sys.stdout)
            ]
    )    
    ## get raw file list based on file range (todo, based on header ?)
    raw_file_list_t1 = make_runlist_allfiles(tel = 1,
                                             year=year,
                                             month=month,
                                             day=day)

    raw_file_list_t2 = make_runlist_allfiles(tel = 2,
                                             year=year,
                                             month=month,
                                             day=day)
 
    ## refine file list by target in header
    if (len(raw_file_list_t1)==0) and (len(raw_file_list_t2)==0):
        logging.warning("NO DATA FOUND : daily analysis ended")
        return

    if (len(raw_file_list_t1)==0) or (len(raw_file_list_t2)==0):
        stereo = False

    datedir = os.path.join(ana_dir,
                           '{:04d}{:02d}{:02d}'.format(year,month,day) )
    Path(datedir).mkdir(exist_ok=True)
    shutil.copy(daily_config_file, datedir)
    shutil.copy(config_file, datedir)

    dict_list_t1 = refine_file_list(raw_file_list_t1)
    dict_list_t2 = refine_file_list(raw_file_list_t2)

    target_list = np.unique(list(dict_list_t1.keys())+list(dict_list_t2.keys()))
    logging.info("Target found ::")
    for target in target_list:
        logging.info('{} :'.format(target))
        try :
            logging.info('CS1 : {}'.format(len(dict_list_t1[target]) ))
        except :
           logging.warning('no CS1 data')
        try :
            logging.info('CS2 : {}'.format(len(dict_list_t2[target]) ))
        except :
           logging.warning('no CS2 data')

     
    for target in target_list:
        if target in ['DARK','Transition','UNKNOWN','BIAS','WRtest']:
            continue
        target_dir = os.path.join(datedir,
                                  '{}'.format(target) )
        Path(target_dir).mkdir(exist_ok=True)

        ## yes, this have a low probability to work!
        ## TODO get target pos.. How ? 
        ## is ok, we don't realy use this for now
        try:
            target_pos = SkyCoord.from_name(target)
        except:
            logging.info("Cant guess the target position!")
            target_pos = SkyCoord(0*u.deg,0*u.deg)

        #CS1
        if cs1 or stereo:
            cs1_dir = os.path.join(target_dir,'cs1')
            Path(cs1_dir).mkdir(exist_ok=True)

            cs1_dl1_dir = os.path.join(cs1_dir,'dl1')
            Path(cs1_dl1_dir).mkdir(exist_ok=True)

            cs1_dl2_dir = os.path.join(cs1_dir,'dl2')
            Path(cs1_dl2_dir).mkdir(exist_ok=True)

            cs1_dl3_dir = os.path.join(cs1_dir,'dl3')
            Path(cs1_dl3_dir).mkdir(exist_ok=True)

        ## CS2
        if cs2 or stereo:
            cs2_dir = os.path.join(target_dir,'cs2')
            Path(cs2_dir).mkdir(exist_ok=True)

            cs2_dl1_dir = os.path.join(cs2_dir,'dl1')
            Path(cs2_dl1_dir).mkdir(exist_ok=True)

            cs2_dl2_dir = os.path.join(cs2_dir,'dl2')
            Path(cs2_dl2_dir).mkdir(exist_ok=True)

            cs2_dl3_dir = os.path.join(cs2_dir,'dl3')
            Path(cs2_dl3_dir).mkdir(exist_ok=True)


        ## STEREO
        if stereo:
            stereo_dir = os.path.join(target_dir,'stereo')
            Path(stereo_dir).mkdir(exist_ok=True)

            stereo_dl1_dir = os.path.join(stereo_dir,'dl1')
            Path(stereo_dl1_dir).mkdir(exist_ok=True)

            stereo_dl2_dir = os.path.join(stereo_dir,'dl2')
            Path(stereo_dl2_dir).mkdir(exist_ok=True)

            stereo_dl3_dir = os.path.join(stereo_dir,'dl3')
            Path(stereo_dl3_dir).mkdir(exist_ok=True)

        ## R0 -> DL1
        # cs1
        aargs = iargs()
        
        aargs.config_file = config_file

        if cs1 and (target in list(dict_list_t1.keys()) ):
            aargs.out_dir     = cs1_dl1_dir
            

            pool = mp.Pool(n_proc)
            pool_results = pool.map(r0_dl1_1file,
                                args_maker(aargs,
                                           dict_list_t1[target]))
            pool.close()

        # cs2
        if cs2 and (target in list(dict_list_t2.keys())):
            aargs.out_dir     = cs2_dl1_dir

            pool = mp.Pool(n_proc)
            pool_results = pool.map(r0_dl1_1file,
                                args_maker(aargs,
                                           dict_list_t2[target]))
            pool.close()

        ## DL1 -> DL2
        aargs.models_dir  = mono_model_dir
        #cs1
        if cs1:
            
            aargs.out_dir     = cs1_dl2_dir

            pool = mp.Pool(n_proc)
            pool_results = pool.map(dl1_dl2_1file_1tel,
                                args_maker(aargs,
                                           glob.glob(cs1_dl1_dir+'/*.h5')))
            pool.close()

            ## DL3
            try:
                aargs.input_dir   = cs1_dl2_dir
                aargs.out_dir     = cs1_dl3_dir
                aargs.target_name = target
                aargs.target_ra = target_pos.ra.to_value(u.deg)
                aargs.target_dec = target_pos.dec.to_value(u.deg)
                aargs.irf_dir = daily_config["irf_dir"]
                dl2_dl3_1dir(aargs)
            except:
                logging.error("DL2 > DL3 failed")

        # cs2
        if cs2:
            aargs.out_dir     = cs2_dl2_dir

            pool = mp.Pool(n_proc)
            pool_results = pool.map(dl1_dl2_1file_1tel,
                                args_maker(aargs,
                                           glob.glob(cs2_dl1_dir+'/*.h5')))
            pool.close()
            ## DL3
            try:
                aargs.input_dir   = c2_dl2_dir
                aargs.out_dir     = cs2_dl3_dir
                aargs.target_name = target
                aargs.target_ra = target_pos.ra.to_value(u.deg)
                aargs.target_dec = target_pos.dec.to_value(u.deg)
                aargs.irf_dir = daily_config["irf_dir"]
                dl2_dl3_1dir(aargs)
            except:
                logging.error("DL2 > DL3 failed")

        ## STEREO
        if stereo:
            aargs.out_dir     = stereo_dl1_dir
            aargs.tel2_dir    = cs2_dl1_dir
            pool = mp.Pool(n_proc)
            pool_results = pool.map(dl1_dl1_1file,
                                    args_maker(aargs,
                                               glob.glob(cs1_dl1_dir+'/*.h5')))
            pool.close()       

            aargs.out_dir   = stereo_dl2_dir
            aargs.models_dir = stereo_model_dir

            pool = mp.Pool(n_proc)
            pool_results = pool.map(dl1_dl2_1file,
                                    args_maker(aargs,
                                               glob.glob(stereo_dl1_dir+'/*.h5')))
            pool.close()
            ## DL3
            try : 
                aargs.input_dir   = stereo_dl2_dir
                aargs.out_dir     = stereo_dl3_dir
                aargs.target_name = target
                aargs.target_ra = target_pos.ra.to_value(u.deg)
                aargs.target_dec = target_pos.dec.to_value(u.deg)
                aargs.irf_dir = daily_config["irf_dir"]
                dl2_dl3_1dir(aargs)
            except:
                logging.error("DL2 > DL3 failed")



    logging.info("Daily analysis ended")

if __name__ == '__main__':

    #os.environ['GAMMAPY_DATA'] = "/data/work/analysis/Daily_analysis/daily_data/"

    if len(sys.argv)>1:
        daily_config_file = sys.argv[1]
    else:
        daily_config_file = DEFAULT_CONFIG_FILE

    with open(daily_config_file) as json_data_file:
        daily_config = json.load(json_data_file)

    run_daily_ana(daily_config_file)




