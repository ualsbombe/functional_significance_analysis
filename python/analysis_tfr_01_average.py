#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:32:14 2022

@author: lau
"""

from config import (fname, submitting_method, tfr_freqs, tfr_n_cycles,
                    tfr_n_jobs,
                    tfr_tmin, tfr_tmax, collapsed_event_id)

from sys import argv
from helper_functions import should_we_run, collapse_event_id

import mne

def this_function(subject, date, overwrite):
    

                        
    output_name = fname.tfr_average(subject=subject,
                                                    date=date,
                                                    tmin=tfr_tmin,
                                                    tmax=tfr_tmax)
    
    if should_we_run(output_name, overwrite):

        epochs = mne.read_epochs(fname.tfr_epochs(
            subject=subject, date=date, tmin=tfr_tmin, tmax=tfr_tmax),
            proj=True)
        tfrs = list()
        
        for event in epochs.event_id:
            tfr = mne.time_frequency.tfr_multitaper(epochs[event],
                                                          tfr_freqs,
                                                          tfr_n_cycles,
                                                          return_itc=False,
                                                          n_jobs=tfr_n_jobs)
            tfr.comment = event
            tfrs.append(tfr)
            
        ## combined values
        epochs = collapse_event_id(epochs, collapsed_event_id)
        for event in epochs.event_id:
            if epochs.event_id[event] > 80: # trials with a response
                tfr = mne.time_frequency.tfr_multitaper(epochs[event],
                                                            tfr_freqs,
                                                            tfr_n_cycles,
                                                            return_itc=False,
                                                            n_jobs=tfr_n_jobs)
                tfr.comment = event
                tfrs.append(tfr)
            
        mne.time_frequency.write_tfrs(output_name, tfrs, overwrite)
            
if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'tave'
    n_jobs = tfr_n_jobs
    deps = ['eve', 'tepo']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))            