#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:58:04 2023

@author: lau
"""

#%% FIGURE 2 - evoked responses and beamformed

from config import fname, recordings, bad_subjects
from manuscript_config import figure_path, fig_2_settings
from manuscript_helper_functions import (load_data_stat_evoked,
                                         set_rc_params, run_stats_evoked,
                                         plot_vertices_evoked,
                                         plot_whole_brain_evoked,
                                         get_difference_vertices_evoked,
                                         plot_difference_vertices_evoked)
import mne
from os.path import join
import numpy as np

#%% LOAD

src_path = fname.anatomy_volumetric_source_space(subject='fsaverage', 
                                                  spacing=7.5)
src = mne.read_source_spaces(src_path)


#%% EVOKED RESPONSES


def plot_evoked(subject, date, events):
    
    set_rc_params(font_size=10)
    
    ga_path = fname.evoked_grand_average_proj_interpolated(
        subject=subject,
        date=date,
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000)
    
    evokeds = mne.read_evokeds(ga_path)
    for evoked in evokeds:
        evoked.crop(tmax=0.400)
    
    titles = ['First Stimulation',
              'Second Stimulation',
              'Weak Stimulation (No jitter)',
              'Weak Stimulation (Jitter)']
    for event_index, event in enumerate(events):
    
        for evoked in evokeds:
            n_characters = len(event)
            if event == evoked.comment[-n_characters:]:
                print(evoked.comment)
                break
    
        fig = evoked.plot(picks='mag',
                              titles=dict(mag=titles[event_index]),
                              time_unit='ms', ylim=dict(mag=(-125, 125))
                              , hline=[0])
        # fig.set_size_inches(8, 6)
        fig.savefig(join(figure_path, 'fig2_' + event  + '_evoked.png'),
                    dpi=300)
        fig = evoked.plot_topomap(ch_type='mag', times=(0.050, 0.124),
                                vlim=(-60, 60), time_unit='ms')
        fig.savefig(join(figure_path, 'fig2_' + event + '_topo.png'), dpi=300)
    

    
plot_evoked('fsaverage', '20210825_000000', ['s1', 's2', 'w0', 'w15'])


#%% BEAMFORMER

contrasts = ['s1_s2', 'w0_w15', 'o0_o15']

#%% load data
data_evoked = dict()
for contrast_index, contrast in enumerate(contrasts):
    data_evoked[contrast] = load_data_stat_evoked(
                                fig_2_settings['combinations'][contrast_index],
                                recordings,
                                bad_subjects,
                                fig_2_settings['n_layers'],
                                [])
    
#%% run stats

stats = dict()
for contrast in contrasts:
    stats[contrast] = dict()

max_vertices_dict = run_stats_evoked(stats, data_evoked,
                                      fig_2_settings['combinations'],
                                      fig_2_settings['label_dict'])

#%% plotting time courses

## s1 vs s2 and w0 vs w15
contrasts = ['s1_s2', 'w0_w15']
rois = ['SI', 
        # 'CL6', 'SII',
        'CL1']
times = [0.050, 0.097]
for contrast_index, contrast in enumerate(contrasts):
    for roi_index, roi in enumerate(rois):
        time = times[roi_index]
        print(contrast + ': ' + roi)
        plot_vertices_evoked(
            fig_2_settings['combinations'][contrast_index]['contrast'],
            max_vertices_dict[contrast], roi, ylim=(0, 65e-15),
            cluster_times=stats[contrast][roi], time_ms=int(time*1e3))


#%% plotting whole brains


rois = ['SI',
        'CL6', 'SII', 'CL1'
        ]
times = [0.050,
            # 0.088, 0.120, 
            0.097
         ]

for roi, time in zip(rois, times):
    plot_whole_brain_evoked('fsaverage', '20210825_000000',
                            fig_2_settings['combinations'][0]['contrast'][0],
                            fig_2_settings['weight_norm'],
                            fig_2_settings['n_layers'],
                            max_vertices_dict, time, roi, 's1_s2',
                            src)


#%% SUPPLEMENTARIES FOR FIGURE 2

# get differences

times = np.arange(-200, 1001, 1)

## fig 1.

fig = plot_difference_vertices_evoked(times,
                                      get_difference_vertices_evoked('s1_s2',
                                             max_vertices_dict['s1_s2'][0],
                                             'SI', data_evoked),
title='Primary Somatosensory Cortex L\nDifference: First Stimulation and Second Stimulation')

fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig2_difference_s1_s2_SI.png'), dpi=300)

## fig 2.

fig = plot_difference_vertices_evoked(times,
                                      get_difference_vertices_evoked('w0_w15',
                                             max_vertices_dict['s1_s2'][0],
                                             'SI', data_evoked),
title='Primary Somatosensory Cortex L\nDifference: Weak - no jitter and Weak - jitter')

fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig2_difference_w0_w15_SI.png'), dpi=300)

## fig 3.

fig = plot_difference_vertices_evoked(times,
                                      get_difference_vertices_evoked('s1_s2',
                                             max_vertices_dict['s1_s2'][0],
                                             'CL1', data_evoked),
title='Cerebellum Crus 1 L\nDifference: First Stimulation and Second Stimulation')

fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig2_difference_s1_s2_CL1.png'), dpi=300)


## fig 4.
fig = plot_difference_vertices_evoked(times,
                                      get_difference_vertices_evoked('w0_w15',
                                             max_vertices_dict['s1_s2'][0],
                                             'CL1', data_evoked),
title='Cerebellum Crus 1 L\nDifference: Weak - no jitter and Weak - jitter')

fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig2_difference_w0_w15_CL1.png'), dpi=300)

#%% SUPPLEMENTARIES FOR FIGURE 4

times = np.arange(-200, 1001, 1)


fig = plot_difference_vertices_evoked(times,
                                      get_difference_vertices_evoked('s1_s2',
                                             max_vertices_dict['s1_s2'][0],
                                             'CL6', data_evoked),
title='Cerebellum 6 L\nDifference: First Stimulation and Second Stimulation')

fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig4_difference_s1_s2_CL6.png'), dpi=300)


fig = plot_difference_vertices_evoked(times,
                                      get_difference_vertices_evoked('w0_w15',
                                             max_vertices_dict['s1_s2'][0],
                                             'CL6', data_evoked),
title='Cerebellum 6 L\nDifference: Weak - no jitter and Weak - jitter')

fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig4_difference_w0_w15_CL6.png'), dpi=300)