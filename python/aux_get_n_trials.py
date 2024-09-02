#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 16:56:46 2023

@author: lau
"""

#%% IMPORTS

from config import (recordings, fname, evoked_fmax, evoked_fmin, evoked_tmin,
                    evoked_tmax, bad_subjects, bad_channels)
import mne
import numpy as np


#%% N TRIALS

for recording_index, recording in enumerate(recordings):
    subject = recording['subject']
    date = recording['date']
    if subject in bad_subjects:
        continue
    if subject == 'fsaverage':
        continue
    print(subject)
    
    evokeds = mne.read_evokeds(fname.evoked_average_proj(subject=subject,
                                                         date=date,
                                                         tmin=evoked_tmin,
                                                         tmax=evoked_tmax,
                                                         fmin=evoked_fmin,
                                                         fmax=evoked_fmax))
    
    if recording_index == 0:
        n_trials = dict()
        for evoked in evokeds:
            n_trials[evoked.comment] = list()
            
    for evoked in evokeds:
        n_trials[evoked.comment].append(evoked.nave)
        
        
#%%# print means

for evoked in evokeds:
    this_mean = round(np.mean(n_trials[evoked.comment]))
    this_median = np.median(n_trials[evoked.comment])
    this_std = round(np.std(n_trials[evoked.comment]), 2)
    print(evoked.comment + ': Mean: ' + str(this_mean) + \
          '. Median: ' + str(this_median) + '. Std: ' + str(this_std))
        
        
#%% BAD CHANNELS

n_bad_magnetometers = dict()

for recording in recordings:
    subject = recording['subject']
    these_bad_magnetometers = list()
    these_bad_channels = bad_channels[subject]
    for this_bad_channel in these_bad_channels:
        if this_bad_channel[:3] == 'MEG' and this_bad_channel[-1] == '1':
            print(this_bad_channel)
            these_bad_magnetometers.append(this_bad_channel)
    n_bad_magnetometers[subject] = len(these_bad_magnetometers)
    


for n_bad in [0, 1, 2]:
    n_counter = 0
    for subject in n_bad_magnetometers:
        if n_bad_magnetometers[subject] == n_bad:
            n_counter += 1
    print(n_counter)
    
            