#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:30:19 2022

@author: lau
"""

from config import (fname, submitting_method, hilbert_lcmv_contrasts,
                    hilbert_lcmv_regularization,
                    hilbert_tmin, hilbert_tmax,
                    hilbert_fmins, hilbert_fmaxs, 
                    recordings, bad_subjects)
from sys import argv
from helper_functions import should_we_run
import mne

def this_function(subject, date, overwrite):
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        for this_contrast in hilbert_lcmv_contrasts:
            
            first_event = this_contrast[0]
            second_event = this_contrast[1]
            ratio_grand_average = dict() # initialize
            ratio_grand_average['first'] = list()
            ratio_grand_average['second'] = list()
            for event_index, event in enumerate(this_contrast):
                output_name = fname.source_hilbert_beamformer_grand_average(
                    subject=subject, date=date,
                    fmin=fmin, fmax=fmax,
                    tmin=hilbert_tmin, tmax=hilbert_tmax,
                    reg=hilbert_lcmv_regularization,
                    event=event,
                    first_event=first_event, 
                    second_event=second_event)    
                ratio_output_name = \
                    fname.source_hilbert_beamformer_contrast_grand_average(
                        subject=subject, date=date,
                        fmin=fmin, fmax=fmax,
                        tmin=hilbert_tmin, tmax=hilbert_tmax,
                        reg=hilbert_lcmv_regularization,
                        event=event,
                        first_event=first_event, 
                        second_event=second_event)
           
                if should_we_run(output_name, overwrite) and \
                    should_we_run(ratio_output_name, overwrite):
                    subject_counter = 0
                    for recording_index, recording in enumerate(recordings):
                        subject_name = recording['subject']
                        if subject_name in bad_subjects:
                            continue # skip the subject
                        subject_counter += 1
                        date = recording['date']
                        
                        lcmv = mne.read_source_estimate(
                            fname.source_hilbert_beamformer_morph(
                                subject=subject_name,date=date,
                                fmin=fmin, fmax=fmax,
                                tmin=hilbert_tmin,
                                tmax=hilbert_tmax,
                                reg=hilbert_lcmv_regularization, event=event,
                                first_event=first_event,
                                second_event=second_event) + '-stc.h5')
                
                        ## single grand averages
                        if recording_index == 0:
                            grand_average = lcmv.copy()
                        else:
                            grand_average._data += lcmv.data
                        ## ratio grand average
                        if event_index == 0:
                            first_lcmv = lcmv.copy()
                            ratio_grand_average['first'].append(first_lcmv)
                            print(len(ratio_grand_average['first']))
                        elif event_index == 1:
                            second_lcmv = lcmv.copy()
                            print(len(ratio_grand_average['second']))
                            ratio_grand_average['second'].append(second_lcmv)
                

                            
                            
                    grand_average._data /= subject_counter # get the mean
                    grand_average.save(output_name, ftype='h5')
            
            
            subject_counter = 0
            for recording_index, recording in enumerate(recordings):
                subject_name = recording['subject']
                if subject_name in bad_subjects:
                    continue # skip the subject
                first_lcmv = \
                    ratio_grand_average['first'][subject_counter]
                second_lcmv = \
                    ratio_grand_average['second'][subject_counter]
                if recording_index == 0:
                    ratio = first_lcmv.copy()
                    ratio._data = \
                        (first_lcmv.data - second_lcmv.data) / \
                        (first_lcmv.data + second_lcmv.data)
                else:
                    this_ratio = \
                        (first_lcmv.data - second_lcmv.data) / \
                        (first_lcmv.data + second_lcmv.data)
                    ratio._data += this_ratio
                subject_counter += 1
    
                    
            ratio._data /= subject_counter
                

            ratio.save(ratio_output_name, ftype='h5')
                
if submitting_method == 'hyades_frontend':
    queue = 'highmem_short.q'
    job_name = 'hlcmvga'
    n_jobs = 2
    deps = ['eve', 'hfilt', 'hepo', 'have', 'mri', 'ana', 'fwd', 'hlcmv',
            'mhlcmv']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))