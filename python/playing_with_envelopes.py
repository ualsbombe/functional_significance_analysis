#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:06:30 2021

@author: lau
"""

## read in evokeds
# mne.read_evokeds(fname, proj=False)

evokeds = None

import signal_envelope as se
from scipy.interpolate import interp1d
import numpy as np

for evoked in evokeds:
    n_channels = evoked.info['nchan']
    n_samples = evoked.data.shape[1]
    for channel_index in range(n_channels):
        this_data = evoked.data[channel_index, :]
        envelope_points = se.get_frontiers(this_data, 1)
        end_points = (np.abs(this_data[envelope_points])[0],
                      np.abs(this_data[envelope_points])[-1])
        interpolation_function = interp1d(envelope_points,
                                          np.abs(this_data[envelope_points]),
                                          kind='slinear', fill_value=end_points,
                                          bounds_error=False)
        estimated_envelope = interpolation_function(np.arange(n_samples))
        evoked._data[channel_index, :] = estimated_envelope

for evoked in evokeds:
    evoked.info['bads'] = ['MEG2631']
    # evoked._data = np.abs(evoked.data)
    evoked.pick_types(meg='mag')