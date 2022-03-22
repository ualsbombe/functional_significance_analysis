#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 15:24:43 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_fmin, evoked_fmax,
                    evoked_tmin, evoked_tmax, recordings, bad_subjects)
from sys import argv
from helper_functions import should_we_run
import mne

def this_function(subject, date, overwrite):
    
    output_names = list()
    output_names.append(fname.evoked_grand_average_proj_interpolated(
                                                          subject=subject,
                                                          date=date,
                                                          fmin=evoked_fmin,
                                                          fmax=evoked_fmax,
                                                          tmin=evoked_tmin,
                                                          tmax=evoked_tmax))
    output_names.append(fname.evoked_grand_average_proj(subject=subject,
                                                        date=date,
                                                        fmin=evoked_fmin,
                                                        fmax=evoked_fmax,
                                                        tmin=evoked_tmin,
                                                        tmax=evoked_tmax))
    
    for output_name in output_names:
        if should_we_run(output_name, overwrite):
            grand_average_evokeds = dict()
            ## sort files
            for recording_index, recording in enumerate(recordings):
                subject = recording['subject']
                if subject in bad_subjects:
                    continue ## skip the subject
                date = recording['date']
                evokeds = mne.read_evokeds(fname.evoked_average_proj(
                                    subject=subject,
                                    date=date,
                                    fmin=evoked_fmin,
                                    fmax=evoked_fmax,
                                    tmin=evoked_tmin,
                                    tmax=evoked_tmax))
                
                for evoked in evokeds:
                    event = evoked.comment
                    if recording_index == 0:
                        grand_average_evokeds[event] = [evoked]
                    else:
                        grand_average_evokeds[event].append(evoked)
            
            ## calculate grand averages
            grand_averages = list()
            for grand_average_evoked in grand_average_evokeds:
                these_evokeds = grand_average_evokeds[grand_average_evoked]    
                if 'interpolated' in output_name:
                    grand_average = mne.grand_average(these_evokeds,
                                                      interpolate_bads=True)
                else:
                    grand_average = mne.grand_average(these_evokeds,
                                                      interpolate_bads=False)
                grand_average.comment = grand_average.comment + ': ' + \
                    grand_average_evoked
                grand_averages.append(grand_average)
                
            mne.write_evokeds(output_name, grand_averages)
            
            
if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'egave'
    n_jobs = 1
    deps = ['eve', 'efilt', 'eepo', 'eave']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))