#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:14:49 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_fmin, evoked_fmax,
                    evoked_tmin, evoked_tmax, evoked_baseline, evoked_decim,
                    evoked_event_id, evoked_reject)

from sys import argv
from helper_functions import should_we_run

import mne

def evoked_epochs(subject, date, overwrite):
    
    output_names = list()
    output_names.append(fname.evoked_epochs_no_proj(subject=subject,
                                                    date=date,
                                                    fmin=evoked_fmin,
                                                    fmax=evoked_fmax,
                                                    tmin=evoked_tmin,
                                                    tmax=evoked_tmax))
                        
    output_names.append(fname.evoked_epochs_proj(   subject=subject,
                                                    date=date,
                                                    fmin=evoked_fmin,
                                                    fmax=evoked_fmax,
                                                    tmin=evoked_tmin,
                                                    tmax=evoked_tmax))
    
    
    for output_name in output_names:
        if should_we_run(output_name, overwrite):
            raw = mne.io.read_raw_fif(fname.evoked_filter(subject=subject,
                                                          date=date,
                                                          fmin=evoked_fmin,
                                                          fmax=evoked_fmax))
            events = mne.read_events(fname.events(subject=subject, date=date))
            if 'no_proj' in output_name:
                proj = False
                reject = None
            else:
                proj = True
                reject = evoked_reject
                
            epochs = mne.Epochs(raw, events, evoked_event_id,
                                evoked_tmin, evoked_tmax, evoked_baseline,
                                decim=evoked_decim, reject=reject,
                                proj=proj)
            
            epochs.save(output_name, overwrite=overwrite)
            
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'epo'
    n_jobs = 2

if submitting_method == 'hyades_backend':
    print(argv[:])
    evoked_epochs(subject=argv[1], date=argv[2], overwrite=argv[3])            