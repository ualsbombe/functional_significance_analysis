#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:30:12 2021

@author: lau
"""

from config import submitting_method, fname
from helper_functions import update_links_in_qsub

## make sure that there are symbolic links in the qsub folder, where
## the cluster calls functions from
update_links_in_qsub()

def submit_job(recording, function_name, raw_filename, overwrite):
    subject = recording['subject']
    date = recording['date']
    
    if submitting_method == 'local':   
        if function_name == 'create_folders':
            from analysis_00_create_folders import create_folders
            create_folders(subject, date)
        elif function_name == 'find_events':
            from analysis_01_find_events import find_events
            find_events(subject, date, raw_filename, overwrite)
        elif function_name == 'evoked_filter':
            from analysis_02_evoked_filter import evoked_filter
            evoked_filter(subject, date, overwrite)
        elif function_name == 'evoked_epochs':
            from analysis_03_evoked_epochs import evoked_epochs
            evoked_epochs(subject, date, overwrite)
        elif function_name == 'evoked_average':
            from analysis_04_evoked_average import evoked_average
            evoked_average(subject, date, overwrite)                
        elif function_name == 'hilbert_filter':
            from analysis_05_hilbert_filter import hilbert_filter
            hilbert_filter(subject, date, overwrite)  
        elif function_name == 'hilbert_epochs':
            from analysis_06_hilbert_epochs import hilbert_epochs
            hilbert_epochs(subject, date, overwrite)            
        elif function_name == 'hilbert_average':
            from analysis_07_hilbert_average import hilbert_average
            hilbert_average(subject, date, overwrite)   
        elif function_name == 'source_model':
            from analysis_anatomy_01_source_model import source_model
            source_model(subject, overwrite)                     
        elif function_name == 'forward_model':
            from analysis_anatomy_03_forward_model import forward_model
            forward_model(subject, date, overwrite)              
    
    elif submitting_method == 'hyades_frontend':
        from stormdb.cluster import ClusterJob    
        overwrite_string = str(int(overwrite))
        
        
        ## general
        
        if function_name == 'create_folders':
            from analysis_00_create_folders import queue, job_name, n_jobs
            cmd = "'python' 'analysis_00_create_folders.py' " + \
                  subject + " " + date
                  
        elif function_name == 'find_events':
            from analysis_01_find_events import queue, job_name, n_jobs
            cmd = "'python' 'analysis_01_find_events.py' " + \
                  subject + " " + date + " " + raw_filename + " " + \
                      overwrite_string
        
        ## evoked analysis
        elif function_name == 'evoked_filter':
            from analysis_evoked_00_filter import queue, job_name, n_jobs
            cmd = "'python' 'analysis_evoked_00_filter.py' " + \
                  subject + " " + date + " " + overwrite_string
                  
        elif function_name == 'evoked_epochs':
            from analysis_evoked_01_epochs import queue, job_name, n_jobs
            cmd = "'python' 'analysis_evoked_01_epochs.py' " + \
                  subject + " " + date + " " + overwrite_string  
                  
        elif function_name == 'evoked_average':
            from analysis_evoked_02_average import queue, job_name, n_jobs
            cmd = "'python' 'analysis_evoked_02_average.py' " + \
                  subject + " " + date + " " + overwrite_string  
                  
        ## hilbert analysis
        elif function_name == 'hilbert_filter':
            from analysis_hilbert_00_filter import queue, job_name, n_jobs
            cmd = "'python' 'analysis_hilbert_00_filter.py' " + \
                  subject + " " + date + " " + overwrite_string
                  
        elif function_name == 'hilbert_epochs':
            from analysis_hilbert_01_epochs import queue, job_name, n_jobs
            cmd = "'python' 'analysis_hilbert_01_epochs.py' " + \
                  subject + " " + date + " " + overwrite_string
                  
        elif function_name == 'hilbert_average':
            from analysis_hilbert_02_average import queue, job_name, n_jobs
            cmd = "'python' 'analysis_hilbert_02_average.py' " + \
                  subject + " " + date + " " + overwrite_string
    
        ## anatomy
        elif function_name == 'import_reconstruct_watershed':
            pass ## use bash
        
        elif function_name == 'source_model':
            from analysis_anatomy_01_source_model import (queue, job_name,
                                                          n_jobs)
            cmd = "'python' 'analysis_anatomy_01_source_model.py' " + \
                  subject + " " + overwrite_string
                  
        elif function_name == 'scalp_surfaces':
            pass ## use bash
                  
        elif function_name == 'forward_model':
            from analysis_anatomy_03_forward_model import (queue, job_name,
                                                          n_jobs)
            cmd = "'python' 'analysis_anatomy_03_forward_model.py' " + \
                  subject + " " + date + " " + overwrite_string
                  
        ## source evoked   
        
        elif function_name == 'evoked_beamformer':
            from analysis_source_evoked_00_beamformer_contrast import (queue,
                                                job_name, n_jobs)
            cmd = "'python' " + \
                "'analysis_source_evoked_00_beamformer_contrast.py' " + \
                    subject + " " + date + " " + overwrite_string
                  
        ## source hilbert
        
        elif function_name == 'hilbert_beamformer':
            from analysis_source_hilbert_00_beamformer_contrast import (queue,
                                                job_name, n_jobs)
            cmd = "'python' " + \
                "'analysis_source_hilbert_00_beamformer_contrast.py' " + \
                    subject + " " + date + " " + overwrite_string
                  
                  
        cj = ClusterJob(cmd=cmd,
                        queue=queue,
                        job_name=job_name + subject,
                        proj_name='MINDLAB2021_MEG-CerebellarClock-FuncSig',
                        working_dir=fname.python_qsub_path,
                        n_threads=n_jobs)
        cj.submit()
        
    else:
        print(submitting_method)
        raise RuntimeError('Unspecified "submitting_method"')
    