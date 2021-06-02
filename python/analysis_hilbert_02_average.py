#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 15:55:44 2021

@author: lau
"""

from config import (fname, submitting_method, hilbert_fmins, hilbert_fmaxs,
                    hilbert_tmin, hilbert_tmax, hilbert_collapsed_event_id)

from sys import argv
from helper_functions import should_we_run

import mne

def hilbert_average(subject, date, overwrite):
    
    for hilbert_fmin, hilbert_fmax in zip(hilbert_fmins, hilbert_fmaxs):
    
        output_names = list()
        output_names.append(fname.hilbert_average_no_proj(subject=subject,
                                                        date=date,
                                                        fmin=hilbert_fmin,
                                                        fmax=hilbert_fmax,
                                                        tmin=hilbert_tmin,
                                                        tmax=hilbert_tmax))
                            
        output_names.append(fname.hilbert_average_proj( subject=subject,
                                                        date=date,
                                                        fmin=hilbert_fmin,
                                                        fmax=hilbert_fmax,
                                                        tmin=hilbert_tmin,
                                                        tmax=hilbert_tmax))
        
        for output_name in output_names:
            if should_we_run(output_name, overwrite):
                if 'no_proj' in output_name:
                    epochs = mne.read_epochs(fname.hilbert_epochs_no_proj(
                        subject=subject, date=date, fmin=hilbert_fmin,
                         fmax=hilbert_fmax, tmin=hilbert_tmin,
                         tmax=hilbert_tmax),
                        proj=False)
                else:
                    epochs = mne.read_epochs(fname.hilbert_epochs_proj(
                        subject=subject, date=date, fmin=hilbert_fmin,
                         fmax=hilbert_fmax, tmin=hilbert_tmin, 
                         tmax=hilbert_tmax),
                        proj=True)
                hilbert_evokeds = list()
                
                for event in epochs.event_id:
                    hilbert_evokeds.append(epochs[event].average())
                    
                ## combined values
                for collapsed_event_id in hilbert_collapsed_event_id:
                    mne.epochs.combine_event_ids(epochs,
             hilbert_collapsed_event_id[collapsed_event_id]['old_event_ids'],
             hilbert_collapsed_event_id[collapsed_event_id]['new_event_id'],
                                             copy=False)
                for event in epochs.event_id:
                    if epochs.event_id[event] > 80: # trials with a response
                        hilbert_evokeds.append(epochs[event].average())                    
                    
                mne.write_evokeds(output_name, hilbert_evokeds)
                hilbert_evokeds = None
            
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'have'
    n_jobs = 3

if submitting_method == 'hyades_backend':
    print(argv[:])
    hilbert_average(subject=argv[1], date=argv[2],
                    overwrite=bool(int(argv[3])))