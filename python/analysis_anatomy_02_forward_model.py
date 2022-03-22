#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:00:03 2021

@author: lau
"""

from config import (fname, submitting_method, bem_ico, src_spacing,
                    split_recording_subjects)
from sys import argv
from helper_functions import should_we_run, read_split_raw_info

import mne

def this_function(subject, date, overwrite):
    
    output_name = fname.anatomy_forward_model(subject=subject, date=date,
                                              spacing=src_spacing)
    
    if should_we_run(output_name, overwrite):
        if subject in split_recording_subjects:
            info = read_split_raw_info(subject, date)
        else:
            info = mne.io.read_info(fname.raw_file(subject=subject, date=date))
        trans = fname.anatomy_transformation(subject=subject, date=date)
        src = fname.anatomy_volumetric_source_space(subject=subject,
                                                    spacing=src_spacing)
        bem = fname.anatomy_bem_solutions(subject=subject, ico=bem_ico)
        
        fwd = mne.make_forward_solution(info, trans, src, bem)
        mne.write_forward_solution(output_name, fwd, overwrite)
        
if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'fwd'
    n_jobs = 1
    deps = ['mri', 'ana']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))           