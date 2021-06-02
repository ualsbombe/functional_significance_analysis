#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  7 16:37:48 2021

@author: lau
"""

from config import fname, submitting_method, evoked_fmin, evoked_fmax
from sys import argv
from helper_functions import should_we_run

import mne

def evoked_filter(subject, date, overwrite):

    
    output_name = fname.evoked_filter(subject=subject, date=date,
                                      fmin=evoked_fmin, fmax=evoked_fmax)
    # figure_name = fname.events_plot(raw_filename=raw_filename, subject=subject,
                                    # date=date)
    if should_we_run(output_name, overwrite):
        raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                 date=date), preload=True)
    
        if evoked_fmin is None:
            l_freq = None
            h_freq = evoked_fmax
        elif evoked_fmax is None:
            l_freq = evoked_fmin
            h_freq = None
        else:
            l_freq = evoked_fmin
            h_freq = evoked_fmax
        
        raw.filter(l_freq, h_freq)
        raw.save(output_name, overwrite=overwrite)

if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'efilt'
    n_jobs = 2

if submitting_method == 'hyades_backend':
    print(argv[:])
    evoked_filter(subject=argv[1], date=argv[2], overwrite=argv[3])