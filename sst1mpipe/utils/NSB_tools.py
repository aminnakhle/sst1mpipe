#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created Feb 29 2024
"""

import numpy as np
import astropy.units as u
from scipy import interpolate

## spline aprox NSB : VAR[ADC]->SHIFT[ADC]
## these are rough aprox.. To be updated!

TCK1 = (np.array([0.        ,  0.        , 43.65192388, 62.89172074, 67.91267773,
                  72.96316339, 73.42825723, 73.88641097, 73.88641097]),
        np.array([-4.33760412, 29.75088614, 53.56764605, 64.52243001, 79.39786228,
                  85.76278027, 89.85345909,  0.        ,  0.        ]),
        1)


TCK2=(np.array([0.        ,   0.        ,  93.74735497, 126.46232902,
                181.54405469, 256.80398702, 266.62290877, 292.98812266,
                294.71402901, 295.67072519, 298.04051867, 339.91032956,
                339.91032956]),
      np.array([ -1.74764142,  48.41280122,  68.36843193, 104.62272377,
                168.14318275, 179.34289593, 212.07400565, 214.10474285,
                215.17017626, 217.7828114 , 268.19128994,   0.        ,
                  0.        ]),
      1)




def VAR_to_shift(baseline_VAR,ntel):
    if ntel==21:
        shift = interpolate.splev(baseline_VAR, TCK1, der=0)
    elif ntel==22:
        shift = interpolate.splev(baseline_VAR, TCK2, der=0)
    else:
        print("ERROR ntel should be 21 or 22")
        shift=0
    return shift

## linear aproximation of the gain+XT+PDE variation 
## as a function of the baseline shift

def shift_to_I_modifier_t1(B_shift,slope = -0.0077):
    return slope*B_shift+1

def shift_to_I_modifier_t2(B_shift,slope = -0.0019):
    return slope*B_shift+1


def VAR_to_Idrop(baseline_VAR,ntel):
    ## Usage :
    ## I_corr = I / I_drop
    if ntel==21:
        shift = interpolate.splev(baseline_VAR, TCK1, der=0)
        I_drop = shift_to_I_modifier_t1(shift)
    elif ntel==22:
        shift = interpolate.splev(baseline_VAR, TCK2, der=0)
        I_drop = shift_to_I_modifier_t2(shift)
    else:
        print("ERROR ntel should be 21 or 22")
    return I_drop

def get_optical_eff_shift(ntel):
    ## Usage :
    ## I_corr = I * optical_eff_shift

    if ntel==21:
        optical_eff_shift = 0.96

    elif ntel==22:
        optical_eff_shift = 1.1

    else:
        print("ERROR ntel should be 21 or 22")
    return optical_eff_shift 


def get_simple_nsbrate(bs_shift,dc_to_pe,binlenght=4*u.ns):
    rate = bs_shift/dc_to_pe / binlenght
    return rate.to('MHz')

def gain_drop_th(nsb_rate, cell_capacitance=85. * u.fF, bias_resistance=2.4 * u.kohm):
    return 1 - 1 / (1 + (nsb_rate * cell_capacitance * bias_resistance).to_value(1))

def VAR_to_NSB(baseline_VAR,ntel,dc_to_pe=None):
    if (ntel==21) and (dc_to_pe==None):
        no_nsb_f = 21.8
    elif (ntel==22) and (dc_to_pe==None):
        no_nsb_f = 25.5
    else:
        no_nsb_f=dc_to_pe
    NSB_rate = get_simple_nsbrate(VAR_to_shift(baseline_VAR,ntel),
                                  no_nsb_f * VAR_to_Idrop(baseline_VAR,ntel))
    return NSB_rate
    
    
