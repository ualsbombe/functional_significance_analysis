#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:59:38 2023

@author: lau
"""

#%% FIGURE 4 - omission oscillations

from config import fname, recordings, bad_subjects
from manuscript_helper_functions import (find_label_vertices,
                                         plot_full_cluster,
                                         save_T1_plot_only_hilbert,
                                         plot_vertices_hilbert,
                                         get_combination,
                                         get_cluster_times_label,
                                         get_difference_vertices_hilbert,
                                         plot_difference_vertices_hilbert,
                                         load_data_hilbert,
                                         get_difference_vertices_hilbert,
                                         plot_difference_vertices_hilbert
                                         )
from manuscript_config import fig_4_settings, figure_path
import mne
from os.path import join
import numpy as np
    
    
#%% get label indices

src_path = fname.anatomy_volumetric_source_space(subject='fsaverage', 
                                                      spacing=7.5)
src = mne.read_source_spaces(src_path)

stc_indices, src_indices = find_label_vertices(fig_4_settings['label_dict'],
                                                src)


#%% plot full cluster and vertices


rois = [
        # 'SI', 
        'CL6',
        # 'CL6_R'
        # 'SII',
        # 'CL1'
        ]

for roi in rois:

    fig, stats, time = plot_full_cluster('o0_o15',
                          fig_4_settings['fmin'],
                          fig_4_settings['fmax'],
                          fig_4_settings['stat_tmin'],
                          fig_4_settings['stat_tmax'],
                          roi, stc_indices, src_indices, src)
    T1_filename = join(figure_path, 'fig4_' + roi + '_hilbert.png')
    save_T1_plot_only_hilbert(fig, T1_filename, time)
    
    cluster_times = get_cluster_times_label(stats, stc_indices, roi)
    plot_vertices_hilbert(get_combination('o0', 'o15'), src_indices[roi], roi,
                          None, fig_4_settings['fmin'],
                          fig_4_settings['fmax'], 'fig4', cluster_times,
                          time_ms=None)
    
    
#%% SUPPLEMENTARY FIG -LOAD

data_hilbert = load_data_hilbert(fig_4_settings['combinations'][2], recordings,
                                 bad_subjects,
                                     1, 14, 30)
#%% PLOT SUPPLEMENTARY FIGURE 3

times = np.arange(-750, 751, 1)

## fig 1.

fig = plot_difference_vertices_hilbert(times,
                          get_difference_vertices_hilbert(src_indices['SI'],
                                                          data_hilbert),
title='Primary Somatosensory Cortex L\nDifference: Omission - no jitter and Omission - jitter')
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig3_ratio_o0_o15_SI.png'), dpi=300)
## fig 2.

fig = plot_difference_vertices_hilbert(times,
                          get_difference_vertices_hilbert(src_indices['SII'],
                                                          data_hilbert),
title='Secondary Somatosensory Cortex L\nDifference: Omission - no jitter and Omission - jitter')
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig3_ratio_o0_o15_SII.png'), dpi=300)

## fig 3.

fig = plot_difference_vertices_hilbert(times,
                          get_difference_vertices_hilbert(src_indices['CL1'],
                                                          data_hilbert),
title='Cerebellar Crus 1 L\nDifference: Omission - no jitter and Omission - jitter')
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig3_ratio_o0_o15_CL1.png'), dpi=300)

#%% PLOT SUPPLEMENTARY FIGURE 4

times = np.arange(-750, 751, 1)

## fig 1.

fig = plot_difference_vertices_hilbert(times,
                          get_difference_vertices_hilbert(src_indices['CL6'],
                                                          data_hilbert),
title='Cerebellum 6 L\nDifference: Omission - no jitter and Omission - jitter')
fig.set_size_inches(8, 6)
fig.savefig(join(figure_path, 'sfig4_ratio_o0_o15_CL6.png'), dpi=300)