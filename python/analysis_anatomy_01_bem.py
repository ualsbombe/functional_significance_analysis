#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 10:49:23 2022

@author: lau
"""

#heavily inspired by Chris Bailey
#https://github.com/meeg-cfin/notebooks/blob/master/forward_modelling/Generate%20BEM%20surfaces%20from%20MR.ipynb

from config import (fname, submitting_method, stls, bem_conductivities)
from sys import argv
import mne
from helper_functions import should_we_run, run_process_and_write_output
from os import makedirs, listdir
from os.path import join

## simnibs/freesurfer commands
# meshfix
# mris_transform

def this_function(subject, date, overwrite):
    
    subjects_dir = fname.simnibs_freesurfer_subjects_dir
    ## create bem directory
    makedirs(fname.simnibs_bem_path(subject=subject), exist_ok=True)
    ## link to stl files
    for stl in stls:
        target = join(fname.subject_m2m_path(subject=subject), stl)
        link_name = join(fname.simnibs_bem_path(subject=subject), stl)
    
        cmd = [
                'ln', '-sf',
                target,
                link_name            
              ]
        run_process_and_write_output(cmd, subjects_dir)
    ## link 'subjects_dir' to 'fs_****_dirs'
    target = fname.subject_fs_path(subject=subject)
    link_name = join(subjects_dir, subject)
    
    cmd = [
            'ln', '-sf',
            target,
            link_name
          ]
    run_process_and_write_output(cmd, subjects_dir)
    
    ## fix meshes and transform to the correct space (Chris Bailey's code)
    xfm_volume = join(fname.subject_m2m_path(subject=subject), 'tmp', 
                      'subcortical_FS.nii.gz') ## why this one?
    xfm = join(fname.subject_m2m_path(subject=subject), 'tmp', 'unity.xfm')
    bem_surfaces = dict(
                        inner_skull=stls[0], ## csf.stl
                        outer_skull=stls[1], ## skull.stl
                        outer_skin =stls[2]  ## skin.stl
                        ) 
    for bem_layer, surf in bem_surfaces.items():
        surf_fname = join(fname.simnibs_bem_path(subject=subject), surf)
        bem_fname  = join(fname.simnibs_bem_path(subject=subject), bem_layer)
        
        filenames = listdir(fname.simnibs_bem_path(subject=subject))
        if 'inner_skull.surf' not in filenames or \
           'outer_skull.surf' not in filenames or \
           'outer_skin.surf'  not in filenames:
            ## fix mesh
            print('im here')
            cmd = [
                    'meshfix',
                    surf_fname, ## input
                    '-u', '10', '--vertices', '5120', '--fsmesh', ## options
                    '-o', bem_fname ## output
                  ]
            run_process_and_write_output(cmd, subjects_dir)
            
            ## transform
            cmd = [
                    'mris_transform',
                    '--dst', xfm_volume,
                    '--src', xfm_volume,
                    bem_fname + '.fsmesh',
                    xfm,
                    bem_fname + '.surf'
                  ]
            run_process_and_write_output(cmd, subjects_dir)
        
    ## create bem_model
    ## FIXME: create loop instead
    output_name = fname.anatomy_simnibs_bem_surfaces(subject=subject,
                                                     n_layers=3)
    if should_we_run(output_name, overwrite):
        bem_surfaces = mne.bem.make_bem_model(
                    subject, ico=None,
                    conductivity=bem_conductivities[1], ## 3-layer
                    subjects_dir=subjects_dir)
        mne.bem.write_bem_surfaces(output_name, bem_surfaces, overwrite)
   
    output_name = fname.anatomy_simnibs_bem_surfaces(subject=subject,
                                                     n_layers=1)
    if should_we_run(output_name, overwrite):
        bem_surfaces = mne.bem.make_bem_model(
                    subject, ico=None,
                    conductivity=bem_conductivities[0], ## 1-layer
                    subjects_dir=subjects_dir)
        mne.bem.write_bem_surfaces(output_name, bem_surfaces, overwrite)
    
    ## bem_solution 
    output_name = fname.anatomy_simnibs_bem_solutions(subject=subject,
                                                      n_layers=1)
    if should_we_run(output_name, overwrite):
        input_name = fname.anatomy_simnibs_bem_surfaces(subject=subject,
                                                        n_layers=1)
        bem_surfaces = mne.bem.read_bem_surfaces(input_name)
        bem_solution = mne.bem.make_bem_solution(bem_surfaces)
        mne.bem.write_bem_solution(output_name, bem_solution, overwrite)
        
        ## bem_solution 
    output_name = fname.anatomy_simnibs_bem_solutions(subject=subject,
                                                      n_layers=3)
    if should_we_run(output_name, overwrite):
        input_name = fname.anatomy_simnibs_bem_surfaces(subject=subject,
                                                        n_layers=3)
        bem_surfaces = mne.bem.read_bem_surfaces(input_name)
        bem_solution = mne.bem.make_bem_solution(bem_surfaces)
        mne.bem.write_bem_solution(output_name, bem_solution, overwrite)
        
        
    

if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'snbem'
    n_jobs = 1
    deps = ['snibs']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2],
                  overwrite=bool(int(argv[3])))
    