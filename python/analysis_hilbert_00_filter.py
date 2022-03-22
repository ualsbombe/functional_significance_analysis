#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:39:43 2021

@author: lau
"""

from config import (fname, submitting_method, hilbert_fmins, hilbert_fmaxs,
                    split_recording_subjects, bad_channels)
from sys import argv
from helper_functions import should_we_run, read_split_raw

import mne

def this_function(subject, date, overwrite):

    
    for hilbert_fmin, hilbert_fmax in zip(hilbert_fmins, hilbert_fmaxs):

        output_name = fname.hilbert_filter(subject=subject, date=date,
                                          fmin=hilbert_fmin, fmax=hilbert_fmax)
    

        if should_we_run(output_name, overwrite):
            if subject in split_recording_subjects:
                raw = read_split_raw(subject, date)
            else:
                raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                     date=date), preload=True)
            
            raw.info['bads'] = bad_channels[subject]
            
            raw.filter(hilbert_fmin, hilbert_fmax)
            raw.save(output_name, overwrite=overwrite)

if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'hfilt'
    n_jobs = 4
    deps = None

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))