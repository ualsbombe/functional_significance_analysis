#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:31:10 2022

@author: lau
"""

from config import (fname, submitting_method, split_recording_subjects,
                    tfr_tmin, tfr_tmax, tfr_baseline, tfr_decim,
                    tfr_event_id, tfr_reject)

from sys import argv
from helper_functions import (should_we_run, check_if_all_events_present,
                              read_split_raw)

import mne

def this_function(subject, date, overwrite):
    
                       
    output_name = fname.tfr_epochs(subject=subject,
                                    date=date,
                                    tmin=tfr_tmin,
                                    tmax=tfr_tmax)

    
    if should_we_run(output_name, overwrite):
        if subject in split_recording_subjects:
            raw = read_split_raw(subject, date)
        else:
            raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                     date=date))
        events = mne.read_events(fname.events(subject=subject, date=date))
        event_id = check_if_all_events_present(events, tfr_event_id)

        proj = True
        reject = tfr_reject
            
        epochs = mne.Epochs(raw, events, event_id,
                            tfr_tmin, tfr_tmax, tfr_baseline,
                            decim=tfr_decim, reject=reject,
                            proj=proj)
        
        epochs.save(output_name, overwrite=overwrite)
            
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'tepo'
    n_jobs = 2
    deps = ['eve']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))            