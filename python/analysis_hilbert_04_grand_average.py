#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 14:57:08 2021

@author: lau
"""

from config import(fname, submitting_method, hilbert_fmins, hilbert_fmaxs,
                   hilbert_tmin, hilbert_tmax, recordings, bad_subjects)
from sys import argv
from helper_functions import should_we_run
import mne

def this_function(subject, date, overwrite):
    
    for hilbert_fmin, hilbert_fmax in zip(hilbert_fmins, hilbert_fmaxs):
        
        output_names = list()
        # output_names.append(fname.hilbert_grand_average_no_proj(
        #                                             subject=subject,
        #                                             date=date,
        #                                             fmin=hilbert_fmin,
        #                                             fmax=hilbert_fmax,
        #                                             tmin=hilbert_tmin,
        #                                             tmax=hilbert_tmax))
        output_names.append(fname.hilbert_grand_average_proj(
                                                    subject=subject,
                                                    date=date,
                                                    fmin=hilbert_fmin,
                                                    fmax=hilbert_fmax,
                                                    tmin=hilbert_tmin,
                                                    tmax=hilbert_tmax))
        
        for output_name in output_names:
            if should_we_run(output_name, overwrite):
                grand_average_z_contrasts = dict()
                ## sort files
                for recording_index, recording in enumerate(recordings):
                    this_subject = recording['subject']
                    if this_subject in bad_subjects:
                        continue ## skip the subject
                    date = recording['date']
                    if 'no_proj' in output_name:
                        proj = False
                        input_name = fname.hilbert_wilcoxon_no_proj(
                                                        subject=this_subject,
                                                        date=date,
                                                        fmin=hilbert_fmin,
                                                        fmax=hilbert_fmax,
                                                        tmin=hilbert_tmin,
                                                        tmax=hilbert_tmax)
                    else:
                        proj = True
                        input_name = fname.hilbert_wilcoxon_proj(
                                                        subject=this_subject,
                                                        date=date,
                                                        fmin=hilbert_fmin,
                                                        fmax=hilbert_fmax,
                                                        tmin=hilbert_tmin,
                                                        tmax=hilbert_tmax)
                    
                    z_contrasts = mne.read_evokeds(input_name, proj=proj)
                    
                    for z_contrast in z_contrasts:
                        event = z_contrast.comment
                        if recording_index == 0:
                            grand_average_z_contrasts[event] = [z_contrast]
                        else:
                            grand_average_z_contrasts[event].append(z_contrast)
                            
                ## calculate grand averages
                grand_averages = list()
                for grand_average_z_contrast in grand_average_z_contrasts:
                    this_z_contrast = \
                            grand_average_z_contrasts[grand_average_z_contrast]
                    grand_average = mne.grand_average(this_z_contrast,
                                                      interpolate_bads=True)
                    grand_average.comment = grand_average.comment + ':' + \
                        grand_average_z_contrast
                    grand_averages.append(grand_average)

                mne.write_evokeds(output_name, grand_averages)
                
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'hgave'
    n_jobs = 2
    deps = ['eve', 'hfilt', 'hepo', 'have', 'hwil']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))