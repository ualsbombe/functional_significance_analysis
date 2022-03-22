#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 12:34:28 2021

@author: lau
"""

from config import (fname, submitting_method, behavioural_data_time_stamps,
                    split_recording_subjects, subjects_missing_n_trials)
from sys import argv
from helper_functions import should_we_run, read_split_raw

import mne
import numpy as np


def this_function(subject, date, overwrite):
    output_name = fname.events(subject=subject, date=date)
    figure_name = fname.events_plot(subject=subject, date=date)
    if should_we_run(output_name, overwrite):

        if subject in split_recording_subjects:
            raw = read_split_raw(subject, date)
        else:
            raw = mne.io.read_raw_fif(fname.raw_file(subject=subject,
                                                     date=date))
            
        subject_code = raw.info['subject_info']['last_name'] + '_' + \
                       raw.info['subject_info']['first_name']
        date_short = date[:-7]
        date_short = date_short[:4] + '_' + date_short[4:6] + '_' + \
            date_short[6:]
        time_stamp = behavioural_data_time_stamps[subject]
        events = mne.find_events(raw, min_duration=0.002)
        
        behavioural_data_file = fname.behavioural_data(subject=subject,
                                                    date=date,
                                                    subject_code=subject_code,
                                                    date_short=date_short,
                                                    time_stamp=time_stamp)
        
        triggers = np.genfromtxt(behavioural_data_file, skip_header=1,
                                 usecols=0, delimiter=',')
        responses = np.genfromtxt(behavioural_data_file, skip_header=1,
                                  usecols=2, delimiter=',', dtype=str)
        
        if subject in subjects_missing_n_trials:
            n_trials = subjects_missing_n_trials[subject]
            triggers = triggers[n_trials:]
            responses = responses[n_trials:]
        
        for trigger_index, trigger in enumerate(triggers):
            ## 2**8 is correct; 2**9 is incorrect
            if trigger > 80:
                this_trigger = trigger.copy() # to not modify in place
                this_response = responses[trigger_index]
                if trigger == 81 or trigger == 97: ## weak event
                    if this_response == 'yes':
                        new_coding = this_trigger + 2**8 ## hit
                    elif this_response == 'no':
                        new_coding = this_trigger + 2**9 ## miss
                                
                if trigger == 144 or trigger == 160: ## omission event
                    if this_response == 'yes':
                        new_coding = this_trigger + 2**9 ## false alarm
                    elif this_response == 'no':
                        new_coding = this_trigger + 2**8 ## correct rejection
            
                new_event = np.expand_dims(events[trigger_index, :],
                                            axis=0)
                new_event[0, 2] = new_coding ## modify in place
        
        mne.write_events(output_name, events)
        fig = mne.viz.plot_events(events)
        fig.savefig(figure_name)
        
        
        
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'eve'
    n_jobs = 2
    deps = None

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))