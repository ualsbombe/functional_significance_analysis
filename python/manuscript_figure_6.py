#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:00:59 2023

@author: lau
"""

#%% FIGURE 6 - envelope correlations

from config import fname, recordings, bad_subjects
from manuscript_config import fig_6_settings, figure_path
from manuscript_helper_functions import (get_data_array, find_label_vertices,
                                         get_t, get_median_stc,
                                         save_T1_plot_only_envelope,
                                         do_anova)
import mne
import numpy as np
from nilearn import datasets
from os.path import join


#%% get data

src = mne.read_source_spaces(fname.anatomy_volumetric_source_space(
                                subject='fsaverage', spacing=7.5))


fig_path = fname.subject_figure_path(subject='fsaverage',
                                      date='20210825_000000')
n_sources = len(src[0]['vertno'])
n_subjects = len(recordings)

data = np.zeros(shape=(n_subjects, n_sources, n_sources))


events = ['w0', 'w15', 'o0', 'o15']
data_dict = dict()
for event in events:

    this_data, these_indices = get_data_array(recordings, bad_subjects,
                                               fig_6_settings['fmin'],
                                               fig_6_settings['fmax'],
                                               fig_6_settings['tmin'],
                                               fig_6_settings['tmax'],
                                               event, data)
    
    data_dict[event] = this_data[these_indices, :, :]
    
    
#%% get all labels

atlas = datasets.fetch_atlas_aal()
labels = atlas['labels']

label_dict = dict()
for label in labels:
    label_dict[label] = dict(label=label, atlas='AAL',
                             restrict_time_index=None)
    
vertex_indices, src_indices = find_label_vertices(label_dict, src)


#%% get t values

ts = dict()
contrasts = [('w0', 'w15'), ('o0', 'o15')]

for contrast in contrasts:
    contrast_name = contrast[0] + '_' + contrast[1]
    ts[contrast_name] = dict()
    
    ts_dict, values = get_t(data_dict[contrast[0]], data_dict[contrast[1]],
                            'Cerebelum_Crus1_L', vertex_indices)
    
    ts[contrast_name]['ts'] = ts_dict
    ts[contrast_name]['values'] = values


#%% plot masked brains

contrasts = [('w0', 'w15'), ('o0', 'o15')]

for contrast in contrasts:
    contrast_name = contrast[0] + '_' + contrast[1]
    median_stc = get_median_stc(data_dict[contrast[0]], data_dict[contrast[1]],
                                'Cerebelum_Crus1_L', vertex_indices, src,
                                ts[contrast_name]['ts'])
    
    fig = median_stc.plot(src, colormap='bwr',
                    clim=dict(kind='value', pos_lims=(0, 0.002, 0.004)),
                    initial_pos=(0.011, -0.018, 0.00))
    
    filename = join(figure_path, 'fig6_' + contrast_name + '_envelope.png')
    save_T1_plot_only_envelope(fig, filename)
    
    
#%% plot anova

do_anova(ts['w0_w15']['values'], ts['o0_o15']['values'],
         'Thalamus_L', figure_path)

#%% print sig areas

from scipy.stats import t
contrasts = [('w0', 'w15'), ('o0', 'o15')]
cutoff = abs(t.ppf(0.025, n_subjects-2))
for contrast in contrasts:
    print(contrast)
    contrast_name = contrast[0] + '_' + contrast[1]
    for key in ts[contrast_name]['ts']:
        if ts[contrast_name]['ts'][key] > cutoff or \
                ts[contrast_name]['ts'][key] < -cutoff:
            pval = t.sf(np.abs(ts[contrast_name]['ts'][key]),
                              n_subjects - 2)*2
            print(key)
            print(round(ts[contrast_name]['ts'][key], 2))
            print(round(pval, 3))
        
