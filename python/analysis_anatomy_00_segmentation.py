#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 09:30:15 2022

@author: lau
"""

## from https://github.com/meeg-cfin/notebooks/blob/master/forward_modelling/Generate%20BEM%20surfaces%20from%20MR.ipynb

from config import (fname, submitting_method,
                    subjects_with_MRs_from_elsewhere, subjects_with_no_T2,
                    t1_file_ending, t2_file_ending)
from os import listdir, chdir
from os.path import join
from helper_functions import run_process_and_write_output
from sys import argv


## commands
# mri_convert dicom to nii.gz
# mri2mesh --all --t2mask <subject> T1.nii.gz T2.nii.gz 

def this_function(subject, date, overwrite):
    if subject in subjects_with_MRs_from_elsewhere:
        mr_path = fname.subject_MR_elsewhere_path(subject=subject, date=date)
    else:
        mr_path = fname.subject_MR_path(subject=subject, date=date)
    directories = listdir(mr_path)
    subjects_dir = fname.simnibs_subjects_dir
    
    ## MRI CONVERT
    output_paths = dict()
    for directory in directories:
        if directory[-len(t1_file_ending):] == t1_file_ending or \
            directory[-len(t2_file_ending):] == t2_file_ending:
            image_path = join(mr_path, directory, 'files')
            image_filename = listdir(image_path)[0]
            full_path = join(image_path, image_filename)
            
            simnibs_path = fname.subject_simnibs_path(subject=subject)
            output_path = join(simnibs_path, directory + '.nii.gz')
            if t1_file_ending in directory:
                output_paths['T1'] = output_path
            if t2_file_ending in directory:
                output_paths['T2'] = output_path

            if len(listdir(simnibs_path)) < 2 or overwrite:            
                command = [
                            'mri_convert',
                            full_path, ## input
                            output_path 
                          ]
                
                run_process_and_write_output(command, subjects_dir)
    if subject in subjects_with_no_T2:
        output_paths['T2'] = '' ## don't add the path below
       
    ## MRI2MESH
    if len(listdir(simnibs_path)) < 3 or overwrite: # only nii.gz's are there 
        chdir(simnibs_path) ## the command below creates subdirectories in
                            ## the current path
        command = [
                    'mri2mesh',
                    '--all',
                    '--t2mask',
                    '--nocleanup', ## keep temporary files
                    subject,
                    output_paths['T1'],
                    output_paths['T2']
                  ]
        
        if subject in subjects_with_no_T2:
            command = [
                        'mri2mesh',
                        '--all',
                        '--nocleanup', ## keep temporary files
                        subject,
                        output_paths['T1']
                      ]
        run_process_and_write_output(command, subjects_dir)

if submitting_method == 'hyades_frontend':
    queue = 'long.q'
    job_name = 'snibs'
    n_jobs = 1
    deps = None

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2],
                  overwrite=bool(int(argv[3])))  