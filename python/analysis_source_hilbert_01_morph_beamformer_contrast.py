#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:15:38 2022

@author: lau
"""

from config import (fname, submitting_method, src_spacing,
                    hilbert_lcmv_contrasts, hilbert_lcmv_regularization,
                    hilbert_tmin, hilbert_tmax, hilbert_fmins, hilbert_fmaxs)
from sys import argv
from helper_functions import should_we_run

import mne

def this_function(subject, date, overwrite):
    morph_name = fname.anatomy_morph_volume(subject=subject,
                                            spacing=src_spacing)
    morph = mne.read_source_morph(morph_name)
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        for this_contrast in hilbert_lcmv_contrasts:
        
            
            input_names = list()
            ## contrast 
            input_names.append(fname.source_hilbert_beamformer_contrast(
                                            subject=subject,
                                            date=date,
                                            fmin=fmin,
                                            fmax=fmax,
                                            tmin=hilbert_tmin,
                                            tmax=hilbert_tmax,
                                            first_event=this_contrast[0],
                                            second_event=this_contrast[1],
                                            reg=hilbert_lcmv_regularization))
            ## first and second events
            for event in this_contrast:
                input_names.append(fname.source_hilbert_beamformer(
                                    subject=subject,
                                    date=date,
                                    fmin=fmin,
                                    fmax=fmax,
                                    tmin=hilbert_tmin,
                                    tmax=hilbert_tmax,
                                    event=event,
                            reg=hilbert_lcmv_regularization,
                            first_event=this_contrast[0],
                            second_event=this_contrast[1])) # because h5
            
            output_names = list()
            ## contrast
            output_names.append(fname.source_hilbert_beamformer_contrast_morph(
                                subject=subject,
                                date=date,
                                fmin=fmin,
                                fmax=fmax,
                                tmin=hilbert_tmin,
                                tmax=hilbert_tmax,
                                first_event=this_contrast[0],
                                second_event=this_contrast[1],
                                reg=hilbert_lcmv_regularization))
            ## first and second events
            for event in this_contrast:
                output_names.append(fname.source_hilbert_beamformer_morph(
                                    subject=subject,
                                    date=date,
                                    fmin=fmin,
                                    fmax=fmax,
                                    tmin=hilbert_tmin,
                                    tmax=hilbert_tmax,
                                    event=event,
                            reg=hilbert_lcmv_regularization,
                            first_event=this_contrast[0],
                            second_event=this_contrast[1]))
            
            
            for name_index, output_name in enumerate(output_names):
                if should_we_run(output_name + '-stc.h5', overwrite):
                    stc = mne.read_source_estimate(input_names[name_index])
                    print(output_name)
                    stc_morph = morph.apply(stc)
                    stc_morph.save(output_name, ftype='h5')
            
if submitting_method == 'hyades_frontend':
    queue = 'highmem_short.q'
    job_name = 'hmlcmv'
    n_jobs = 1
    deps = ['eve', 'hfilt', 'hepo', 'have', 'hlcmv', 'mri', 'ana', 'fwd']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3]))) 