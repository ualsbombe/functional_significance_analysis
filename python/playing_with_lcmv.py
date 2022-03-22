#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 16:15:50 2021

@author: lau
"""

import mne
from config import collapsed_event_id
from helper_functions import collapse_event_id
event = 'o0'
evokeds = mne.read_evokeds('fc-filt-None-40-Hz--0.2-1.0-s-no_proj-ave.fif',
                           proj=False)
epochs = mne.read_epochs('fc-filt-None-40-Hz--0.2-1.0-s-no_proj-epo.fif',
                         preload=False, proj=False)

for evoked in evokeds:
    if evoked.comment == event:
        break
evoked_proj = evoked.copy()
evoked_proj.apply_proj()
evoked_proj.plot()
evoked.del_proj()    
    
epochs = collapse_event_id(epochs, collapsed_event_id)
epochs = epochs[event]
epochs.load_data()
epochs.del_proj()  
epochs.pick_types(meg='mag')
    
info = epochs.info
forward = mne.read_forward_solution('fc-volume-7.5_mm-fwd.fif')
data_cov = mne.compute_covariance(epochs, tmin=0, tmax=None)

filters = mne.beamformer.make_lcmv(info, forward, data_cov, reg=0.00)

lcmv = mne.beamformer.apply_lcmv(evoked, filters)

src = forward['src']

lcmv.plot(src=src)