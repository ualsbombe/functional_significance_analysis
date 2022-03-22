#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:30:12 2021

@author: lau
"""



def submit_job(recording, function_name, overwrite):
    ## make sure that there are symbolic links in the qsub folder, where
    ## the cluster calls functions from
    update_links_in_qsub()
    subject = recording['subject']
    date = recording['date']
    
    if submitting_method == 'local':   
        if function_name == 'create_folders':
            from analysis_00_create_folders import create_folders
            create_folders(subject, date)

        import_statement = 'from ' + function_name + ' import this_function' 
        exec(import_statement, globals()) ## FIXME: is this dangerous
        this_function(subject, date, overwrite)
    
    elif submitting_method == 'hyades_frontend':
        from stormdb.cluster import ClusterJob    
        overwrite_string = str(int(overwrite))
        
        
        import_statement = 'from ' + function_name + ' import ' + \
            'queue, job_name, n_jobs'    
        exec(import_statement, globals())
        
        
        cmd = "python " + function_name + '.py' + " " + \
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
    