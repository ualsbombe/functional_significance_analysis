#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:00:03 2021

@author: lau
"""

from config import (fname, submitting_method, bem_ico, bem_conductivities,
                    src_spacing, split_recording_subjects, n_jobs_forward,
                    morph_subject_to)
from sys import argv
from helper_functions import should_we_run, read_split_raw_info

import mne

def this_function(subject, date, overwrite):
    
    for bem_conductivity in bem_conductivities:
        n_layers = len(bem_conductivity)
        output_names = list()
       
        output_names.append(fname.anatomy_simnibs_forward_model(
                        subject=subject, date=date, spacing=src_spacing,
                        n_layers=n_layers))
        # output_names.append(fname.anatomy_forward_model(subject=subject,
        #                                                 date=date,
        #                                                 spacing=src_spacing,
        #                                                 n_layers=n_layers))
        

        
        for output_name in output_names:
            if should_we_run(output_name, overwrite):
                if subject in split_recording_subjects:
                    info = read_split_raw_info(subject, date)
                else:
                    info = mne.io.read_info(fname.raw_file(subject=subject,
                                                           date=date))
                trans = fname.anatomy_transformation(subject=subject,
                                                     date=date)
                src = fname.anatomy_volumetric_source_space(
                    subject=subject, spacing=src_spacing)
                if 'simnibs' in output_name:
                    bem = fname.anatomy_simnibs_bem_solutions(subject=subject,
                                                             n_layers=n_layers)
                # else:
                #     bem = fname.anatomy_bem_solutions(subject=subject,
                #                                       ico=bem_ico,
                #                                       n_layers=n_layers)
                
                fwd = mne.make_forward_solution(info, trans, src, bem)
                mne.write_forward_solution(output_name, fwd, overwrite)

                if 'simnibs' in output_name:
                    subjects_dir = fname.subjects_dir
                    ## morph to fsaverage
                    output_name = fname.anatomy_simnibs_morph_volume(
                        subject=subject,
                        spacing=src_spacing)
                    if should_we_run(output_name, overwrite):
                        if subject != 'fsaverage':
                            fwd = mne.read_forward_solution(
                                fname.anatomy_simnibs_forward_model(
                                    subject=subject,
                                    date=date,
                                    spacing=src_spacing))
                            src_to_path = \
                                fname.anatomy_volumetric_source_space(
                                    subject=morph_subject_to,
                                    spacing=src_spacing)
                            src_to = mne.read_source_spaces(src_to_path)
                            morph = mne.compute_source_morph(fwd['src'],
                                                    subject,
                                                    subjects_dir=subjects_dir,
                                                    src_to=src_to)
                            morph.save(output_name, overwrite=overwrite)
                ##
        
if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'fwd'
    n_jobs = n_jobs_forward
    deps = ['mri', 'ana', 'snibs', 'snbem']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))           