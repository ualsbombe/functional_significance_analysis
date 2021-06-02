#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:03:17 2021

@author: lau
"""

## go into subject path 0002 has a visual response

import mne
from helper_functions import collapse_event_id
from config import collapsed_event_id
from os import chdir
import matplotlib.pyplot as plt

plt.close('all')

path = '/home/lau/projects/functional_cerebellum/scratch/MEG/' + \
        '0003/20210507_000000'
# path = '/home/lau/projects/functional_cerebellum/scratch/MEG/' + \
#       '0002/20210507_000000'
          
           
chdir(path)           

condition = 's2'


epochs = mne.read_epochs('fc-filt-None-40-Hz--0.2-0.6-s-no_proj-epo.fif',
                         preload=False, proj=False)

if condition == 'o0' or condition == 'o15' or \
    condition == 'w0' or condition == 'w15':
    epochs = collapse_event_id(epochs, collapsed_event_id)

this_evoked = mne.read_evokeds('fc-filt-None-40-Hz--0.2-0.6-s-no_proj-ave.fif',
                               proj=False, condition=condition)
subject = this_evoked.info['subject_info']['last_name']
if subject == '0002':
    epochs.info['bads'] = ['MEG1531']
    this_evoked.info['bads'] = ['MEG1531']
elif subject == '0003':
    epochs.info['bads'] = ['MEG2631']
    this_evoked.info['bads'] = ['MEG2631']
    
fwd = mne.read_forward_solution('fc-volume-7.5_mm-fwd.fif')

this_evoked.pick_types(meg='mag')

this_evoked_proj = this_evoked.copy()
this_evoked_proj.apply_proj()
this_evoked_proj.plot()
this_evoked.del_proj()

these_epochs = epochs[condition]
these_epochs.del_proj()
these_epochs.load_data()
these_epochs.pick_types(meg='mag')

data_cov = mne.compute_covariance(these_epochs)
data_cov.plot(these_epochs.info)

filters = mne.beamformer.make_lcmv(these_epochs.info, fwd, data_cov,
                                   weight_norm='unit-noise-gain')

lcmv = mne.beamformer.apply_lcmv(this_evoked, filters)

src = fwd['src']

lcmv.plot(src, subject)
