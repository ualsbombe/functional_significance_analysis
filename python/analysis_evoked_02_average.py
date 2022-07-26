#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:55:17 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_fmin, evoked_fmax,
                    evoked_tmin, evoked_tmax, collapsed_event_id)

from sys import argv
from helper_functions import should_we_run, collapse_event_id

import mne

def this_function(subject, date, overwrite):
    
    output_names = list()
    output_names.append(fname.evoked_average_no_proj(subject=subject,
                                                    date=date,
                                                    fmin=evoked_fmin,
                                                    fmax=evoked_fmax,
                                                    tmin=evoked_tmin,
                                                    tmax=evoked_tmax))
                        
    output_names.append(fname.evoked_average_proj(subject=subject,
                                                    date=date,
                                                    fmin=evoked_fmin,
                                                    fmax=evoked_fmax,
                                                    tmin=evoked_tmin,
                                                    tmax=evoked_tmax))
    
    for output_name in output_names:
        if should_we_run(output_name, overwrite):
            if 'no_proj' in output_name:
                epochs = mne.read_epochs(fname.evoked_epochs_no_proj(
                    subject=subject, date=date, fmin=evoked_fmin,
                     fmax=evoked_fmax, tmin=evoked_tmin, tmax=evoked_tmax),
                    proj=False)
            else:
                epochs = mne.read_epochs(fname.evoked_epochs_proj(
                    subject=subject, date=date, fmin=evoked_fmin,
                     fmax=evoked_fmax, tmin=evoked_tmin, tmax=evoked_tmax),
                    proj=True)
            evokeds = list()
            
            for event in epochs.event_id:
                evokeds.append(epochs[event].average())
                
            ## combined values
            epochs = collapse_event_id(epochs, collapsed_event_id)
            for event in epochs.event_id:
                if epochs.event_id[event] > 80: # trials with a response
                    evokeds.append(epochs[event].average())
                
            mne.write_evokeds(output_name, evokeds, overwrite=overwrite)
            
if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'eave'
    n_jobs = 3
    deps = ['eve', 'efilt', 'eepo']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))            