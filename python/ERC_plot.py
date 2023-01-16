#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 16 13:17:23 2022

@author: lau
"""

meg_path = '/home/lau/projects/functional_cerebellum/scratch/MEG/' + \
    '0003/20210802_000000/beamformer_evoked'
mr_path = '/home/lau/projects/functional_cerebellum/scratch/freesurfer/' + \
    'fsaverage/bem'
    
import mne
from os.path import join

# stc = mne.read_source_estimate(join(meg_path,
# 'fc-filt-None-40-Hz--0.2-1.0-s-reg-0.0-s1-filter-s1-s2' + \
#     '-unit-gain-simnibs-n_layers-3-morph-vl-stc.h5'))
    
stc = mne.read_source_estimate(join(meg_path,
    'fc-filt-None-40-Hz--0.2-1.0-s-reg-0.0-w0-filter-w0-w15' + \
        '-unit-gain-simnibs-n_layers-3-morph-vl-stc.h5'))
    
src = mne.read_source_spaces(join(mr_path, 'volume-7.5_mm-src.fif'))

stc.plot(src, initial_pos=(-0.019, -0.052, -0.035), initial_time=0.072)

evokeds = mne.read_evokeds(join(meg_path, '..', 
                                'fc-filt-None-40-Hz--0.2-1.0-s-proj-ave.fif'))
evoked = evokeds[0]
evoked.plot_topomap(times=0.078)

trans = join(meg_path, '..', 'fc-trans.fif')

map_helmet = mne.make_field_map(evoked, trans)
map_head   = mne.make_field_map(evoked, trans=trans, subject='0003',
                                meg_surf='head', n_jobs=7)
map_head_fsaverage = mne.make_field_map(evoked, trans=trans,
                                        subject='fsaverage',
                                meg_surf='head', n_jobs=7)


## potential cerebellar
mne.viz.plot_evoked_field(evoked, map_helmet, time=0.078)
mne.viz.plot_evoked_field(evoked, map_head, time=0.078)
mne.viz.plot_evoked_field(evoked, map_head_fsaverage, time=0.078)

## SII activation

mne.viz.plot_evoked_field(evoked, map_helmet, time=0.128)
mne.viz.plot_evoked_field(evoked, map_head, time=0.128)
mne.viz.plot_evoked_field(evoked, map_head_fsaverage, time=0.128)