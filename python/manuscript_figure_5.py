#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:00:27 2023

@author: lau
"""

#%% FIGURE 5 - stimulation oscillations#

from config import fname, recordings, bad_subjects
from manuscript_helper_functions import (find_label_vertices,
                                         plot_full_cluster,
                                         save_T1_plot_only_hilbert,
                                         plot_vertices_hilbert,
                                         get_combination,
                                         get_cluster_times_label,
                                         load_data_hilbert,
                                         get_difference_vertices_hilbert,
                                         plot_difference_vertices_hilbert)
from manuscript_config import fig_5_settings, figure_path
import mne
from os.path import join
import numpy as np
    
    
#%% get label indices

src_path = fname.anatomy_volumetric_source_space(subject='fsaverage', 
                                                     spacing=7.5)
src = mne.read_source_spaces(src_path)

stc_indices, src_indices = find_label_vertices(fig_5_settings['label_dict'],
                                               src)


#%% plot full cluster and vertices


rois = [
        'SI',
        'SII', 
        'TH', 
        # 'TH_R'
        ]

peak_times = [47, 67, 83]

for roi_index, roi in enumerate(rois):

    peak_time = peak_times[roi_index] ## careful these are gotten from visual
    ## inspection of the first stim peaks from "plot vertices hilbert below"
    fig, stats, time = plot_full_cluster('s1_s2',
                          fig_5_settings['fmin'],
                          fig_5_settings['fmax'],
                          fig_5_settings['stat_tmin'],
                          fig_5_settings['stat_tmax'],
                          roi, stc_indices, src_indices, src,
                          lims=(0.019, 0.021, 0.028))
    T1_filename = join(figure_path, 'fig5_' + roi + '_hilbert.png')
    save_T1_plot_only_hilbert(fig, T1_filename, time)
    
    cluster_times = get_cluster_times_label(stats, stc_indices, roi)
    plot_vertices_hilbert(get_combination('s1', 's2'), src_indices[roi], roi,
                          None, fig_5_settings['fmin'],
                          fig_5_settings['fmax'], 'fig5', cluster_times,
                          time_ms=peak_time)
    
#%% SUPPLEMENTARY FIG

data_hilbert = load_data_hilbert(fig_5_settings['combinations'][0], recordings,
                                 bad_subjects,
                                     1, 14, 30)


#%% PLOT

times = np.arange(-750, 751, 1)

## fig 1.

fig = plot_difference_vertices_hilbert(times,
                          get_difference_vertices_hilbert(src_indices['SI'],
                                                          data_hilbert),
title='Primary Somatosensory Cortex L\nDifference: First Stimulation and Second Stimulation')
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig4_ratio_s1_s2_SI.png'), dpi=300)
## fig 2.

fig = plot_difference_vertices_hilbert(times,
                          get_difference_vertices_hilbert(src_indices['SII'],
                                                          data_hilbert),
title='Secondary Somatosensory Cortex L\nDifference: First Stimulation and Second Stimulation')
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig4_ratio_s1_s2_SII.png'), dpi=300)

## fig 3.

fig = plot_difference_vertices_hilbert(times,
                          get_difference_vertices_hilbert(src_indices['TH'],
                                                          data_hilbert),
title='Thalamus L\nDifference: First Stimulation and Second Stimulation')
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig4_ratio_s1_s2_TH.png'), dpi=300)