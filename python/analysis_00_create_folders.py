#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 15:53:33 2021

@author: lau
"""

from config import fname, submitting_method
from os import makedirs
from sys import argv


def this_function(subject, date, overwrite):
    if overwrite:
        exist_ok=True
    else:
        exist_ok=False
    makedirs(fname.subject_path(subject=subject,
                                date=date), exist_ok=exist_ok)
    makedirs(fname.subject_figure_path(subject=subject,
                                date=date), exist_ok=exist_ok)
    makedirs(fname.subject_behaviour_path(subject=subject,
                                          date=date), exist_ok=exist_ok)
    makedirs(fname.subject_beamformer_hilbert_path(subject=subject, date=date),
                                                    exist_ok=exist_ok)
    makedirs(fname.subject_beamformer_evoked_path(subject=subject, date=date),
                                                     exist_ok=exist_ok)


if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'cf'
    n_jobs = 1
    deps = None

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))