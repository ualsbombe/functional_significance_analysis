#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:01:38 2023

@author: lau
"""

#%% FIGURE 7 - thalamic backup

import numpy as np
from config import fname
from manuscript_config import figure_path
from manuscript_helper_functions import set_rc_params, plot_peak_time_courses
import mne
from os.path import join

#%% evoked - thalamus

channels = [
'MEG0611', 'MEG1011', 'MEG1021', 'MEG0821', 'MEG0941', 'MEG0931', 'MEG0641',
 'MEG0621', 'MEG1031', 'MEG1241', 'MEG1111', 'MEG0741', 'MEG0731', 'MEG2211',
'MEG1831', 'MEG2241', 'MEG2231', 'MEG2011', 'MEG2021', 'MEG2311']

def create_topo_mask(evoked, channels, typ):
    mask = np.zeros(evoked.data.shape)
    for channel in channels:
        this_index = evoked.info.ch_names.index(channel)
        if typ == 'mag':
            mask[this_index, :] = 1
        elif typ == 'grad':
            mask[this_index + 1, :] = 1
        else:
            raise ValueError('"typ" must be "mag" or "grad"')
        
    return mask


    
ga_path = fname.evoked_grand_average_proj_interpolated(subject='fsaverage',
                                                       date='20210825_000000',
                                                       fmin=None, fmax=40,
                                                       tmin=-0.200, tmax=1.000)

evoked = mne.read_evokeds(ga_path, condition='Grand average (n = 28): s1')
evoked.crop(tmax=0.400)
    

set_rc_params(font_size=12)

mag_mask = create_topo_mask(evoked.copy(), channels, 'mag')
grad_mask = create_topo_mask(evoked.copy(), channels, 'grad')

fig = evoked.plot_topomap(ch_type='mag',
                               times=(0.060, 0.084, 0.124, 0.140, 0.170),
                               vlim=(-60, 60),
                               time_unit='ms', mask=mag_mask,
                               mask_params=dict(markersize=8),
                               show_names=False) 
fig = evoked.plot_topomap(ch_type='grad',
                               times=(0.060, 0.084, 0.124, 0.140, 0.170),
                                vlim=(0, 20),
                               # vlim=(0, 8),
                               time_unit='ms', mask=grad_mask,
                               mask_params=dict(markersize=8),
                               show_names=False) 


fig = evoked.plot(picks='mag',
                      titles=dict(mag='Magnetometers:\nFirst Stimulation'),
                      time_unit='ms', ylim=dict(mag=(-125, 125)), hline=[0],
                      xlim=(-50, 200), highlight=(80, 90))
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'fig7_s1_thalamus_highlight_evoked.png'),
            dpi=300)


fig = evoked.plot_topomap(ch_type='mag',
                               times=(0.060, 0.084, 0.124, 0.140, 0.170),
                               vlim=(-60, 60),
                               time_unit='ms', mask=mag_mask,
                               mask_params=dict(markersize=8),
                               show_names=False) 
fig.savefig(join(figure_path, 'fig7_s1_thalamus_highlight_topo.png'), dpi=300)

fig = evoked.plot(picks='grad', 
                  titles=dict(grad='Gradiometers:\nFirst Stimulation'),
                      time_unit='ms', ylim=dict(grad=(-50, 50)), hline=[0],
                      xlim=(-50, 200), highlight=(80, 90))
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'fig7_s1_thalamus_highlight_evoked_grad.png'),
            dpi=300)
fig = evoked.plot_topomap(ch_type='grad',
                               times=(0.060, 0.084, 0.124, 0.140, 0.170),
                               vlim=(0, 20),
                               time_unit='ms', mask=grad_mask,
                               mask_params=dict(markersize=8),
                               show_names=False) 
fig.savefig(join(figure_path, 'fig7_s1_thalamus_highlight_topo_grad.png'),
            dpi=300)


#%% beamformer thalamus

full_path = fname.source_hilbert_beamformer_grand_average(
        subject='fsaverage',
        date='20210825_000000',
        fmin=14, fmax=30,
        tmin=-0.750, tmax=0.750,
        event='s1',
        first_event='s1',
        second_event='s2',
        reg=0.00, weight_norm='unit-noise-gain-invariant',
        n_layers=1)

stc = mne.read_source_estimate(full_path)

set_rc_params(line_width=3)

plot_dict = dict(
                    SI=dict(pos=(-0.017, -0.035, 0.071), vertex=12061),
                    SI_R=dict(pos=(0.017, -0.035, 0.071), vertex=12065),
                    SII=dict(pos=(-0.044, -0.032, 0.019), vertex=8354),
                    SII_R=dict(pos=(0.044, -0.032, 0.019), vertex=8366),
                    TH=dict(pos=(-0.010, -0.019, 0.007), vertex=7140),
                    TH_R=dict(pos=(0.010, -0.019, 0.007), vertex=7142)
                )


plot_peak_time_courses(stc, plot_dict, figure_path, save=True)

