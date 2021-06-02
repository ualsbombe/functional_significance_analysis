#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:39:43 2021

@author: lau
"""

from config import fname, submitting_method, hilbert_fmins, hilbert_fmaxs
from sys import argv
from helper_functions import should_we_run

import mne

def hilbert_filter(subject, date, overwrite):

    
    for hilbert_fmin, hilbert_fmax in zip(hilbert_fmins, hilbert_fmaxs):

        output_name = fname.hilbert_filter(subject=subject, date=date,
                                          fmin=hilbert_fmin, fmax=hilbert_fmax)
    

        if should_we_run(output_name, overwrite):
            raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                     date=date), preload=True)
    
            
            raw.filter(hilbert_fmin, hilbert_fmax)
            raw.save(output_name, overwrite=overwrite)

if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'hfilt'
    n_jobs = 2

if submitting_method == 'hyades_backend':
    print(argv[:])
    hilbert_filter(subject=argv[1], date=argv[2], overwrite=argv[3])