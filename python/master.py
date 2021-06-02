#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 08:37:13 2021

@author: lau
"""

#%% IMPORTS

from submit_jobs import submit_job

#%% RECORDINGS

recordings = [
                  # dict(subject='0001', date='20210416_000000'),
                  # dict(subject='0002', date='20210423_000000'),
                    dict(subject='0002', date='20210507_000000'),
                    dict(subject='0003', date='20210507_000000')
                    # dict(subject='fsaverage', date=None)
             ]

mr_recordings = [
                    dict(subject='0003', date='')
                ]


#%% RUN ALL JOBS

for recording in recordings:
    ## GENERAL
    # submit_job(recording, 'create_folders', None, False)
    # submit_job(recording, 'find_events', 'trigger_test', False)
    # submit_job(recording, 'find_events', 'func_cerebellum', True)
    
    ## EVOKED ANALYSIS
    # submit_job(recording, 'evoked_filter', None,  False)
    # submit_job(recording, 'evoked_epochs', None,  False)
    # submit_job(recording, 'evoked_average', None, False)
    
    ## HILBERT ANALYSIS
    # submit_job(recording, 'hilbert_filter', None,  False)
    # submit_job(recording, 'hilbert_epochs', None,  True)
    # submit_job(recording, 'hilbert_average', None,  True)


    ## ANATOMY PROCESSING
    # submit_job(recording, 'source_model', None, False)  
    # submit_job(recording, 'forward_model', None, False)
    
    ## SOURCE EVOKED
    submit_job(recording, 'evoked_beamformer', None, True)
    
    ## SOURCE HILBERT
    # submit_job(recording, 'hilbert_beamformer', None, False)