#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 10:25:09 2021

@author: lau
"""

from config import (fname, submitting_method, n_jobs_power_spectra, 
                    split_recording_subjects)
from sys import argv
from helper_functions import should_we_run, read_split_raw

import mne


def this_function(subject, date, overwrite):
    output_name = fname.power_spectra_plot(subject=subject, date=date)
    
    if should_we_run(output_name, overwrite):
        if subject in split_recording_subjects:
            read_split_raw(subject, date)
        else:
            raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                     date=date))
        
        fig = raw.plot_psd(n_jobs=n_jobs_power_spectra)
        fig.savefig(output_name)

if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'plotps'
    n_jobs = 3
    deps = None

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))