#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 09:08:56 2021

@author: lau
"""

from config import (fname, submitting_method, src_spacing, bem_ico,
                    bem_conductivities, morph_subject_to)
from sys import argv
from helper_functions import should_we_run
import mne

def this_function(subject, overwrite):

    subjects_dir = fname.subjects_dir
    
    for bem_conductivity in bem_conductivities:
        n_layers = len(bem_conductivity)
        ## bem surfaces
        output_name = fname.anatomy_bem_surfaces(subject=subject, ico=bem_ico,
                                                 n_layers=n_layers)
        
        if should_we_run(output_name, overwrite):
            bem_surfaces = mne.bem.make_bem_model(subject, bem_ico,
                                                  bem_conductivity,
                                                  subjects_dir)
            mne.bem.write_bem_surfaces(output_name, bem_surfaces, overwrite)
            
        ## bem solution
        output_name = fname.anatomy_bem_solutions(subject=subject, ico=bem_ico,
                                                  n_layers=n_layers)
        if should_we_run(output_name, overwrite):
            bem_solution = mne.bem.make_bem_solution(bem_surfaces)
            mne.bem.write_bem_solution(output_name, bem_solution, overwrite)
        
    ## volumetric source space
    output_name = fname.anatomy_volumetric_source_space(subject=subject,
                                                        spacing=src_spacing)
    if should_we_run(output_name, overwrite):
        bem_path = fname.anatomy_bem_surfaces(subject=subject, ico=bem_ico)
        src = mne.source_space.setup_volume_source_space(subject,
                                                         pos=src_spacing,
                                                         bem=bem_path,
                                                    subjects_dir=subjects_dir)
        mne.source_space.write_source_spaces(output_name, src, overwrite)
        
    ## morph to fsaverage
    output_name = fname.anatomy_morph_volume(subject=subject,
                                             spacing=src_spacing)
    if should_we_run(output_name, overwrite):
        if subject != 'fsaverage':
            src_path = fname.anatomy_volumetric_source_space(subject=subject,
                                                         spacing=src_spacing)
            src_to_path = \
                fname.anatomy_volumetric_source_space(subject=morph_subject_to,
                                                      spacing=src_spacing)
            src_to = mne.read_source_spaces(src_to_path)
            morph = mne.compute_source_morph(src_path, subject,
                                             subjects_dir=subjects_dir,
                                             src_to=src_to)
            morph.save(output_name, overwrite=overwrite)
            
if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'ana'
    n_jobs = 1
    deps = ['mri']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], overwrite=bool(int(argv[3])))          