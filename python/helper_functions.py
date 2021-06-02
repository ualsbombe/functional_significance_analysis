#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:29:38 2021

@author: lau
"""

from config import fname
from os.path import isfile, join
from os import environ, listdir
from subprocess import Popen, PIPE
from sys import stdout
import mne

def should_we_run(save_path, overwrite, suppress_print=False,
                  print_file_to_be_saved=False):
    run_function = overwrite or not isfile(save_path)
    if not run_function:
        if not suppress_print:
            print('not overwriting: ' + save_path)
    if overwrite and isfile(save_path):
        print('Overwriting: ' + save_path)
    if print_file_to_be_saved and not overwrite:
        print('Saving: ' + save_path)
        
    return run_function

def run_process_and_write_output(command, subjects_dir, write_output=True):
    
    environment = environ.copy()
    if subjects_dir is not None:
        environment["SUBJECTS_DIR"] = subjects_dir
    process = Popen(command, stdout=PIPE,
                               env=environment)
    ## write bash output in python console
    if write_output:
        for c in iter(lambda: process.stdout.read(1), b''):
            stdout.write(c.decode('utf-8'))
        
def collapse_event_id(epochs, collapsed_event_id):
    for collapsed_event in collapsed_event_id:

        mne.epochs.combine_event_ids(epochs,
                     collapsed_event_id[collapsed_event]['old_event_ids'],
                     collapsed_event_id[collapsed_event]['new_event_id'],
                                         copy=False)
    return epochs

def update_links_in_qsub():
    python_path = fname.python_path
    qsub_path = fname.python_qsub_path
        
    
    script_names = listdir(python_path)
    for script_name in script_names:
        if 'analysis' in script_name:
            ## clean
            command = ['rm', join(qsub_path, script_name)]
            run_process_and_write_output(command, subjects_dir=None,
                                         write_output=False)
            ## create link
            command = ['ln', '-sf', join(python_path, script_name), 
                 join(qsub_path, script_name)]
            run_process_and_write_output(command, subjects_dir=None,
                                         write_output=False)