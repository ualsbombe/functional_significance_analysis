#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:31:29 2021

@author: lau
"""

from config import (fname, submitting_method, hilbert_fmins, hilbert_fmaxs,
                    hilbert_tmin, hilbert_tmax, hilbert_contrasts)
from sys import argv

import mne

def hilbert_wilcoxon(subject, date, overwrite):
    
    for hilbert_fmin, hilbert_fmax in zip(hilbert_fmins, hilbert_fmaxs):
        
        for contrast in hilbert_contrasts:
        
            output_names = list()
            output_names.append(fname.hilbert_wilcoxon_no_proj())