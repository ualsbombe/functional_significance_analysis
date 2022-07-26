#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 15:15:05 2022

@author: lau
"""

from config import (fname, submitting_method,
                    tfr_tmin, tfr_tmax, recordings, bad_subjects,
                    subjects_with_no_BEM_simnibs)
from sys import argv
from helper_functions import should_we_run
import mne

def this_function(subject, date, overwrite):
    
    output_names = list()
    output_names.append(fname.tfr_grand_average(subject=subject,
                                                        date=date,
                                                        tmin=tfr_tmin,
                                                        tmax=tfr_tmax))
    output_names.append(fname.tfr_grand_average_interpolated(
                                                          subject=subject,
                                                          date=date,
                                                          tmin=tfr_tmin,
                                                          tmax=tfr_tmax))  
    for output_name in output_names:
        if should_we_run(output_name, overwrite):
            grand_average_tfrs = dict()
            ## sort files
            for recording_index, recording in enumerate(recordings):
                subject = recording['subject']
                if subject in bad_subjects or \
                    subject in subjects_with_no_BEM_simnibs:
                    continue ## skip the subject
                date = recording['date']
                tfrs = mne.time_frequency.read_tfrs(fname.tfr_average(
                                    subject=subject,
                                    date=date,
                                    tmin=tfr_tmin,
                                    tmax=tfr_tmax))
                
                for tfr in tfrs:
                    event = tfr.comment
                    if recording_index == 0:
                        grand_average_tfrs[event] = [tfr]
                    else:
                        grand_average_tfrs[event].append(tfr)
            
            ## calculate grand averages
            grand_averages = list()
            for grand_average_tfr in grand_average_tfrs:
                these_tfrs = grand_average_tfrs[grand_average_tfr]    
                if 'interpolated' in output_name:
                    grand_average = mne.grand_average(these_tfrs,
                                                      interpolate_bads=True)
                else:
                    grand_average = mne.grand_average(these_tfrs,
                                                      interpolate_bads=False)
                grand_average.comment = grand_average.comment + ': ' + \
                    grand_average_tfr
                grand_averages.append(grand_average)
                
            mne.time_frequency.write_tfrs(output_name, grand_averages,
                                          overwrite)
            
            
if submitting_method == 'hyades_frontend':
    queue = 'highmem_short.q'
    job_name = 'tgave'
    n_jobs = 12
    deps = ['eve', 'tepo', 'tave']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))