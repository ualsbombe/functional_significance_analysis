#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 23:48:30 2022

@author: lau
"""

# import mne
import numpy as np

## load stc, src and clu yourself

stc = None
clu = None
src = None

#%% plot masked t values

tstc = stc.copy()
tstc.crop(-0.200, 0.200)

t_obs = clu['t_obs']
clusters = clu['clusters']
cluster_p_values = clu['cluster_p_values']

tstc._data = t_obs.T

mstc = tstc.copy()
mstc._data = np.zeros(tstc.shape)

p_threshold = 0.09

sig_indices = np.where(cluster_p_values < p_threshold)[0]

if sig_indices.size != 0:

    for sig_index in sig_indices[:]:
        cluster = clusters[sig_index]
        mstc._data[cluster[1], cluster[0]] = \
            tstc.data[cluster[1], cluster[0]]

mstc.plot(src, initial_time=0)