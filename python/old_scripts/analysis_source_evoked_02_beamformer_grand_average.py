#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 13:56:30 2022

@author: lau
"""

from config import (fname, submitting_method, evoked_lcmv_contrasts,
                    evoked_lcmv_regularization, evoked_lcmv_weight_norms,
                    evoked_tmin, evoked_tmax,
                    evoked_fmin, evoked_fmax, 
                    recordings, bad_subjects, bem_conductivities,
                    subjects_with_no_3_layer_BEM_watershed)
from sys import argv
from helper_functions import should_we_run
import mne

def this_function(subject, date, overwrite):
    for this_contrast in evoked_lcmv_contrasts:
        print(this_contrast)
        for event in this_contrast:
            for evoked_lcmv_weight_norm in evoked_lcmv_weight_norms:
                for bem_conductivity in bem_conductivities:
                    n_layers = len(bem_conductivity)

                    output_name = fname.source_evoked_beamformer_grand_average(
                        subject=subject, date=date, fmin=evoked_fmin, 
                        fmax=evoked_fmax,
                        tmin=evoked_tmin, tmax=evoked_tmax,
                        reg=evoked_lcmv_regularization,
                        event=event,
                        first_event=this_contrast[0],
                        second_event=this_contrast[1],
                        weight_norm=evoked_lcmv_weight_norm,
                        n_layers=n_layers)
                
                    if should_we_run(output_name, overwrite):
                        subject_counter = 0
                        for recording_index,recording in enumerate(recordings):
                            subject_name = recording['subject']
                            if subject_name in bad_subjects or \
        (subject_name in subjects_with_no_3_layer_BEM_watershed and \
         n_layers == 3):
                                continue # skip the subject
                            subject_counter += 1
                            date = recording['date']
                            
                            lcmv = mne.read_source_estimate(
                                fname.source_evoked_beamformer_morph(
                            subject=subject_name,date=date,
                            fmin=evoked_fmin, fmax=evoked_fmax,
                            tmin=evoked_tmin,
                            tmax=evoked_tmax,
                            reg=evoked_lcmv_regularization, event=event,
                            first_event=this_contrast[0],
                            second_event=this_contrast[1],
                            weight_norm=evoked_lcmv_weight_norm,
                            n_layers=n_layers))
                    
                            if recording_index == 0:
                                grand_average = lcmv.copy()
                            else:
                                grand_average._data += lcmv.data
                        grand_average._data /= subject_counter # get the mean
                        grand_average.save(output_name, ftype='h5',
                                           overwrite=overwrite)
                    
if submitting_method == 'hyades_frontend':
    queue = 'highmem_short.q'
    job_name = 'elcmvga'
    n_jobs = 4
    deps = ['eve', 'efilt', 'eepo', 'eave', 'mri', 'ana', 'fwd', 'elcmvlala',
            'emlcmvlala']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))