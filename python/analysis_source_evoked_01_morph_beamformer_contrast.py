#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:34:07 2022

@author: lau
"""


from config import (fname, submitting_method, src_spacing,
                    evoked_lcmv_contrasts, evoked_lcmv_regularization,
                    evoked_tmin, evoked_tmax, evoked_fmin, evoked_fmax,
                    evoked_lcmv_weight_norms, bem_conductivities)
from sys import argv
from helper_functions import should_we_run

import mne

def this_function(subject, date, overwrite):
    # morph_name = fname.anatomy_simnibs_morph_volume(subject=subject,
    #                                         spacing=src_spacing)
    morph_name = fname.anatomy_morph_volume(subject=subject,
                                            spacing=src_spacing)
    morph = mne.read_source_morph(morph_name)
    morph.compute_vol_morph_mat()
    for this_contrast in evoked_lcmv_contrasts:
    
        for evoked_lcmv_weight_norm in evoked_lcmv_weight_norms:
            
            for bem_conductivity in bem_conductivities:
                n_layers = len(bem_conductivity)
                
                input_names = list()
                ## contrast 
                # input_names.append(
                #     fname.source_evoked_beamformer_contrast_simnibs(
                #                         subject=subject,
                #                         date=date,
                #                         fmin=evoked_fmin,
                #                         fmax=evoked_fmax,
                #                         tmin=evoked_tmin,
                #                         tmax=evoked_tmax,
                #                         first_event=this_contrast[0],
                #                         second_event=this_contrast[1],
                #                         reg=evoked_lcmv_regularization,
                #                         weight_norm=evoked_lcmv_weight_norm,
                #                         n_layers=n_layers))
                input_names.append(
                    fname.source_evoked_beamformer_contrast(
                                        subject=subject,
                                        date=date,
                                        fmin=evoked_fmin,
                                        fmax=evoked_fmax,
                                        tmin=evoked_tmin,
                                        tmax=evoked_tmax,
                                        first_event=this_contrast[0],
                                        second_event=this_contrast[1],
                                        reg=evoked_lcmv_regularization,
                                        weight_norm=evoked_lcmv_weight_norm,
                                        n_layers=n_layers))
                ## first and second events
                for event in this_contrast:
                    # input_names.append(fname.source_evoked_beamformer_simnibs(
                    #                     subject=subject,
                    #                     date=date,
                    #                     fmin=evoked_fmin,
                    #                     fmax=evoked_fmax,
                    #                     tmin=evoked_tmin,
                    #                     tmax=evoked_tmax,
                    #                     event=event,
                    #             reg=evoked_lcmv_regularization,
                    #             first_event=this_contrast[0],
                    #             second_event=this_contrast[1],
                    #             weight_norm=evoked_lcmv_weight_norm,
                    #             n_layers=n_layers))
                    
                    input_names.append(fname.source_evoked_beamformer(
                                        subject=subject,
                                        date=date,
                                        fmin=evoked_fmin,
                                        fmax=evoked_fmax,
                                        tmin=evoked_tmin,
                                        tmax=evoked_tmax,
                                        event=event,
                                reg=evoked_lcmv_regularization,
                                first_event=this_contrast[0],
                                second_event=this_contrast[1],
                                weight_norm=evoked_lcmv_weight_norm,
                                n_layers=n_layers))
                
                output_names = list()
                ## contrast
                # output_names.append(
                #     fname.source_evoked_beamformer_contrast_simnibs_morph(
                #                     subject=subject,
                #                     date=date,
                #                     fmin=evoked_fmin,
                #                     fmax=evoked_fmax,
                #                     tmin=evoked_tmin,
                #                     tmax=evoked_tmax,
                #                     first_event=this_contrast[0],
                #                     second_event=this_contrast[1],
                #                     reg=evoked_lcmv_regularization,
                #                     weight_norm=evoked_lcmv_weight_norm,
                #                     n_layers=n_layers))
                output_names.append(
                    fname.source_evoked_beamformer_contrast_morph(
                                    subject=subject,
                                    date=date,
                                    fmin=evoked_fmin,
                                    fmax=evoked_fmax,
                                    tmin=evoked_tmin,
                                    tmax=evoked_tmax,
                                    first_event=this_contrast[0],
                                    second_event=this_contrast[1],
                                    reg=evoked_lcmv_regularization,
                                    weight_norm=evoked_lcmv_weight_norm,
                                    n_layers=n_layers))
                ## first and second events
                for event in this_contrast:
                    # output_names.append(
                    #     fname.source_evoked_beamformer_simnibs_morph(
                    #                     subject=subject,
                    #                     date=date,
                    #                     fmin=evoked_fmin,
                    #                     fmax=evoked_fmax,
                    #                     tmin=evoked_tmin,
                    #                     tmax=evoked_tmax,
                    #                     event=event,
                    #             reg=evoked_lcmv_regularization,
                    #             first_event=this_contrast[0],
                    #             second_event=this_contrast[1],
                    #             weight_norm=evoked_lcmv_weight_norm,
                    #             n_layers=n_layers))
                    
                    output_names.append(
                        fname.source_evoked_beamformer_morph(
                                        subject=subject,
                                        date=date,
                                        fmin=evoked_fmin,
                                        fmax=evoked_fmax,
                                        tmin=evoked_tmin,
                                        tmax=evoked_tmax,
                                        event=event,
                                reg=evoked_lcmv_regularization,
                                first_event=this_contrast[0],
                                second_event=this_contrast[1],
                                weight_norm=evoked_lcmv_weight_norm,
                                n_layers=n_layers))
                
                
                for name_index, output_name in enumerate(output_names):
                    if should_we_run(output_name, overwrite):
                        print(output_name)
                        stc = mne.read_source_estimate(input_names[name_index])
                        stc_morph = morph.apply(stc)
                        stc_morph.save(output_name, ftype='h5',
                                       overwrite=overwrite)
            
if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'emlcmv'
    n_jobs = 1
    deps = ['eve', 'efilt', 'eepo', 'eave', 'elcmvlll', 'mri', 'ana', 'fwd',
            'snibs', 'snbem']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3]))) 