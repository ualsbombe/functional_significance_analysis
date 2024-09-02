#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 09:35:21 2023

@author: lau
"""

from config import fname
import mne
import numpy as np
from manuscript_config import rc_params, figure_path
import matplotlib as mpl
from os.path import exists, join
from nilearn import image, datasets
import matplotlib.pyplot as plt
from cycler import cycler
from scipy.stats import t
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


#%% GENERAL

def save_T1_plot_only_hilbert(fig, T1_filename, time):
    set_rc_params()
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    axis_0 = fig.axes[0]
    extent = \
        axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if time is not None:
        cbar_text = axis_0.get_children()[-2]
        cbar_text.set_position((1.0, 1.5))
        cbar_text.set_text('Power change: (x$_1$ - x$_2$) / (x$_1$ + x$_2$) ' + \
                            str(round(time * 1e3, 1)) + ' ms')
        cbar_text.set_fontsize('medium')
        
        # cbar_text.set_color('white')
        # cbar_text.set_x(0.5)
        # cbar_text.set_y(1.1)
    print(fig.axes[0].get_children())
    fig.savefig(T1_filename, dpi=300, bbox_inches=extent.expanded(1.00, 1.00))
    # print(fig.axes[0].get_children()[-2])

def set_rc_params(font_size=None, font_weight=None, line_width=None,
                  rc_params=rc_params):
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['interactive'] = True # otherwise mpl blocks the figures
    if font_size is None:
        font_size = rc_params['font.size']
    if font_weight is None:
        font_weight = rc_params['font.weight']
    if line_width is None:
        line_width = rc_params['lines.linewidth']
            
    mpl.rcParams['font.size'] = font_size
    mpl.rcParams['font.weight'] = font_weight
    mpl.rcParams['lines.linewidth'] = line_width


## https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

def legend_rewriting(legend):
    if legend == 's1':
        legend = 'First Stimulation'
    elif legend == 's2':
        legend = 'Second Stimulation'
    elif legend == 's3':
        legend = 'Third Stimulation'
    elif legend == 'w0':
        legend = 'Weak - no jitter'
    elif legend == 'w15':
        legend = 'Weak - jitter'
    elif legend == 'o0':
        legend = 'Omission - no jitter'
    elif legend == 'o15':
        legend = 'Omission - jitter'
    else:
        raise NotImplementedError('legend: ' + legend +
                                  ' has not been implemented yet')
    return legend

def roi_rewriting(roi_name):
    if roi_name == 'SI':
        title = 'Primary somatosensory cortex L'
    elif roi_name == 'SI_R':
        title = 'Primary somatosensory cortex R'
    elif roi_name == 'CL6':
        title = 'Cerebellum 6 L'
    elif roi_name == 'CL1':
        title = 'Cerebellum Crus 1 L'
    elif roi_name == 'SII':
        title = 'Secondary somatosensory cortex L'
    elif roi_name == 'SII_R':
        title = 'Secondary somatosensory cortex R'
    elif roi_name == 'PT':
        title = 'Putamen L'
    elif roi_name == 'PI':
        title = 'Inferior parietal cortex L'
    elif roi_name == 'TH':
        title = 'Thalamus L'
    elif roi_name == 'TH_R':
        title = 'Thalamus_R'
    elif roi_name == 'PL':
        title = 'Pallidum L'
    return title

def find_max_vertex(stc, label_dict, stc_full_path, src):
    vertices = src[0]['vertno']
    nifti_full_path = stc_full_path[:-2] + 'nii'
    if not exists(nifti_full_path):
        stc.save_as_volume(nifti_full_path, src, overwrite=True)
    img = image.load_img(nifti_full_path)
    data = np.asanyarray(img.dataobj)
    max_vertices = dict()

    for roi, DICT in label_dict.items():
        if DICT['atlas'] == 'AAL':
            atlas = datasets.fetch_atlas_aal()
        elif DICT['atlas'] == 'harvard':
            atlas = \
                datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
        else:
            raise NameError(roi['atlas'] + ' is not implemented')
            
        atlas_img = image.load_img(atlas['maps'])
        atlas_interpolated = image.resample_to_img(atlas_img, img, 'nearest')
        atlas_interpolated_data = np.asanyarray(atlas_interpolated.dataobj)
        all_labels = atlas['labels']
        for label in all_labels:
            if DICT['label'] == label:
                break
        if DICT['atlas'] == 'harvard':
            label_index = all_labels.index(label)
        elif DICT['atlas'] == 'AAL':
            label_index = int(atlas['indices'][atlas['labels'].index(label)])
        mask = atlas_interpolated_data == label_index
        opposite_mask = ~mask
        this_data = data.copy()
        label_data = np.abs(this_data)
        label_data[opposite_mask, :] = 0
        
        if DICT['restrict_time_index'] is not None:
            x, y, z, t = \
                np.where(
            np.max(label_data[:, :, :, :DICT['restrict_time_index']]) == \
                label_data) 
        else:
            x, y, z, t = np.where(np.max(label_data) == label_data)
                
        ## create all coordinates
        stc_voxels = np.array(
            np.unravel_index(vertices, img.shape[:3], order='F')).T

        coordinate = np.concatenate((np.expand_dims(x, 1),
                                     np.expand_dims(y, 1),
                                     np.expand_dims(z, 1)), axis=1)
        
        for voxel_index, voxel in enumerate(stc_voxels):
            if np.all(voxel == coordinate):
                label_vertex = vertices[voxel_index]
                break
        
        max_vertices[roi] = label_vertex
        max_vertices[roi + '_t'] = np.round(stc.times[t][0], 3)
    return max_vertices

def find_label_vertices(label_dict, src):
    
    stc_full_path = fname.source_hilbert_beamformer_grand_average(
          subject='fsaverage', date='20210825_000000', fmin=14, fmax=30,
          tmin=-0.750, tmax=0.750, reg=0.00,
          event='o0', first_event='o0', second_event='o15',
          weight_norm='unit-noise-gain-invariant', n_layers=1)
    stc = mne.read_source_estimate(stc_full_path)
    vertices = src[0]['vertno']
    nifti_full_path = stc_full_path[:-2] + 'nii'
    if not exists(nifti_full_path):
        stc.save_as_volume(nifti_full_path, src)
    img = image.load_img(nifti_full_path)
    data = np.asanyarray(img.dataobj)
    label_indices = dict()
    src_indices = dict()

    for roi, DICT in label_dict.items():
        print('Getting vertices for label: ' + roi)

        these_vertices = list()

        if DICT['atlas'] == 'AAL':
            atlas = datasets.fetch_atlas_aal()
        elif DICT['atlas'] == 'harvard':
            atlas = \
                datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr0-1mm')
        else:
            raise NameError(roi['atlas'] + ' is not implemented')
            
        atlas_img = image.load_img(atlas['maps'])
        atlas_interpolated = image.resample_to_img(atlas_img, img, 'nearest')
        atlas_interpolated_data = np.asanyarray(atlas_interpolated.dataobj)
        all_labels = atlas['labels']
        for label in all_labels:
            if DICT['label'] == label:
                break
        if DICT['atlas'] == 'harvard':
            label_index = all_labels.index(label)
        elif DICT['atlas'] == 'AAL':
            label_index = int(atlas['indices'][atlas['labels'].index(label)])
        mask = atlas_interpolated_data == label_index
        opposite_mask = ~mask
        this_data = data.copy()
        label_data = np.abs(this_data)
        label_data[opposite_mask, :] = 0
        
        if DICT['restrict_time_index'] is not None:
            x, y, z, t = \
                np.where(
            np.max(label_data[:, :, :, :DICT['restrict_time_index']]) == \
                label_data) 
        else:
            x, y, z = np.where(label_data[:, :, :, 0] > 0)
         
            
        if DICT['atlas'] == 'harvard':
            if DICT['harvard_side'] == 'L':
                boolean_index = x < DICT['boolean_index']
                x = x[boolean_index] ## magic number
                y = y[boolean_index]
                z = z[boolean_index]
            elif DICT['harvard_side'] == 'R':
                boolean_index = x > DICT['boolean_index']
                x = x[boolean_index]
                y = y[boolean_index]
                z = z[boolean_index]
            else:
                raise NameError('harvard side must be specified and must' + \
                                ' be L or R')
        
        ## create all coordinates
        stc_voxels = np.array(
            np.unravel_index(vertices, img.shape[:3], order='F')).T

        coordinates = np.concatenate((np.expand_dims(x, 1),
                                     np.expand_dims(y, 1),
                                     np.expand_dims(z, 1)), axis=1)
        
        for voxel_index, voxel in enumerate(stc_voxels):
            for coordinate in coordinates:
                if np.all(voxel == coordinate):
                    label_vertex = vertices[voxel_index]
                    these_vertices.append(label_vertex)
                    break
        indices = list()
        for this_vertex in these_vertices:
            indices.append(np.where(vertices == this_vertex)[0][0])
        
        label_indices[roi] = indices
        src_indices[roi] = these_vertices

    return label_indices, src_indices

def get_combination(x1, x2):
    
    if x1 == 's1' and x2 == 's2':
        combination = [
                    dict(event='s1', first_event='s1', second_event='s2'),
                    dict(event='s2', first_event='s1', second_event='s2'),
                        ]
        
    if x1 == 's2' and x2 == 's3':
        combination = [
                    dict(event='s2', first_event='s2', second_event='s3'),
                    dict(event='s3', first_event='s2', second_event='s3'),
                        ]
            
    if x1 == 'w0' and x2 == 'w15':
        combination = [
                    dict(event='w0',  first_event='w0', second_event='w15'),
                    dict(event='w15', first_event='w0', second_event='w15'),
                        ]
    
    if x1 == 'o0' and x2 == 'o15':
        combination = [
                    dict(event='o0',  first_event='o0', second_event='o15'),
                    dict(event='o15', first_event='o0', second_event='o15'),
                        ]
        
    return combination


#%% FIG 1

#%% FIG 2

def load_data_stat_evoked(combination, recordings, bad_subjects,
                          n_layers, excluded_subjects):
    print(combination)
    for recording_index, recording in enumerate(recordings):
        subject = recording['subject']
        date = recording['date']
        if subject in bad_subjects or \
            (subject in excluded_subjects and \
             n_layers == 3):
            continue
        print(subject)
        full_path = fname.source_evoked_beamformer_morph(
                subject=subject,
                date=date,
                fmin=None, fmax=40,
                tmin=-0.200, tmax=1.000,
                event=combination['contrast'][0]['event'],
                first_event=combination['contrast'][0]['first_event'],
                second_event=combination['contrast'][0]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=n_layers)
        full_path2 = fname.source_evoked_beamformer_morph(
                subject=subject,
                date=date,
                fmin=None, fmax=40,
                tmin=-0.200, tmax=1.000,
                event=combination['contrast'][1]['event'],
                first_event=combination['contrast'][1]['first_event'],
                second_event=combination['contrast'][1]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=n_layers)
        
        stc = mne.read_source_estimate(full_path)
        stc2 = mne.read_source_estimate(full_path2)
        temp = np.expand_dims(stc.data.copy(), 0)
        temp2 = np.expand_dims(stc2.data.copy(), 0)
        
        if recording_index == 0:
            data = np.array(abs(temp) - abs(temp2)) # should we take abs?
        else:
            data = np.concatenate((data, abs(temp) - abs(temp2)), axis=0)
            
    return data

# stats

def get_clusters(data, tmin, tmax, stc, vertex_index, alpha, adj=None):
    tmin_index = find_nearest(stc.times, tmin)
    tmax_index = find_nearest(stc.times, tmax)
    if type(vertex_index) is not list:
        print(vertex_index)
        stc_index = np.where(stc.vertices[0] == vertex_index)[0][0]
        test_data = data[:, stc_index, tmin_index:tmax_index]
    elif type(vertex_index) is list:
        n_vertices = len(vertex_index)
        stc_index = np.where(stc.vertices[0] == vertex_index[0])[0][0]
        test_data = data[:, stc_index, tmin_index:tmax_index]
        for i in range(1, n_vertices):
            this_stc_index = np.where(stc.vertices[0] == vertex_index[i])[0][0]
            this_test_data = data[:, this_stc_index, tmin_index:tmax_index]
            test_data += this_test_data
        test_data /= n_vertices
    print(test_data.shape)    
    # test_data = np.swapaxes(test_data, 1, 2)
    test_times = stc.times[tmin_index:tmax_index]
    print(test_data.shape)
    t_obs, clusters, cluster_pv, H0 = \
        mne.stats.permutation_cluster_1samp_test(test_data,
                                                     n_permutations=1e4,
                                                 seed=7, adjacency=adj)
    
    cluster_indices = np.where(cluster_pv < alpha)[0]
    sig_cluster_pv = cluster_pv[cluster_indices]
    print(cluster_pv)
    cluster_times = list()
    for cluster_index in cluster_indices:
        cluster_time = test_times[clusters[cluster_index][0]]
        cluster_times.append(cluster_time)
        
    return sig_cluster_pv, cluster_times

def run_stats_evoked(stats, data_evoked, combinations, label_dict):
    max_vertices_dict = dict()
    mr_path = fname.subject_bem_path(subject='fsaverage')
    src = mne.read_source_spaces(join(mr_path, 'volume-7.5_mm-src.fif'))


    for comparison_index, comparison in enumerate(stats):
        event = combinations[comparison_index]['contrast'][0]['event']
        first_event = \
            combinations[comparison_index]['contrast'][0]['first_event']
        second_event = \
            combinations[comparison_index]['contrast'][0]['second_event']
        
        ga_path_1 = fname.source_evoked_beamformer_grand_average(
            subject='fsaverage',
            date='20210825_000000',
            fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
            event=event, first_event=first_event, second_event=second_event,
            weight_norm='unit-noise-gain-invariant',
            n_layers=1)
        print('Reading: ' + ga_path_1)
        stc_1 = mne.read_source_estimate(ga_path_1) 
        ga_path_2 = fname.source_evoked_beamformer_grand_average(
            subject='fsaverage',
            date='20210825_000000',
            fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
            event=second_event, first_event=first_event, second_event=second_event,
            weight_norm='unit-noise-gain-invariant',
            n_layers=1)
        print('Reading: ' + ga_path_2)
        stc_2 = mne.read_source_estimate(ga_path_2)
        
        
                
        # NEW ATTEMPT - averages the two max time courses
        
        max_vertices_1 = find_max_vertex(stc_1, label_dict, 
                                      stc_full_path=ga_path_1, src=src)
        
        max_vertices_2 = find_max_vertex(stc_2, label_dict, 
                                      stc_full_path=ga_path_2, src=src)
        
        max_vertices_dict[comparison] = [max_vertices_1, max_vertices_2]
        
        
        for max_vertex_1, max_vertex_2 in zip(max_vertices_1, max_vertices_2):
            if '_t' in max_vertex_1: #just the timing
                continue
            print('\n' + comparison + ' ' + max_vertex_1 + ' ' +\
                   str(max_vertices_1[max_vertex_1]) + \
                       str(max_vertices_2[max_vertex_2]) + '\n')
            stats[comparison][max_vertex_1] = \
                get_clusters(data_evoked[comparison], 0.000, 0.150, stc_1,
                              [max_vertices_1[max_vertex_1], 
                               max_vertices_2[max_vertex_2]],
                              alpha=0.05)
                
    return max_vertices_dict


## actual plotting


def plot_vertices_evoked(combinations, vertex_dict, roi_name, ylim,
                        cluster_times=None, time_ms=None,
                        save=True,
                        subject='fsaverage',
                        date='20210825_000000',
                        weight_norm='unit-noise-gain-invariant'):
    set_rc_params()
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    legends = list()
    
    these_data = list()
    
    for combination_index, combination in enumerate(combinations):
        ga_path = fname.source_evoked_beamformer_grand_average(
        subject=subject,
        date=date,
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm=weight_norm,
                    n_layers=1)
        stc = mne.read_source_estimate(ga_path)
        if 'w' in combination['event'] or 'o' in combination['event']:
            colours = ['#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f',
                             '#bcbd22', '#17becf']
            mpl.rcParams['axes.prop_cycle'] = \
            cycler('color', colours )
        else:
            cycler('color', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
                             '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
                             '#bcbd22', '#17becf'])


        stc.crop(tmin=-0.050, tmax=0.400) ## cropping
        stc._data = np.abs(stc.data) ## taking abs
        if type(vertex_dict) is dict:
            vertex_index = vertex_dict[roi_name]
            stc_index = np.where(stc.vertices[0] == vertex_index)[0][0]
            data_to_plot = stc.data[stc_index, :]
        elif type(vertex_dict) is list:
            stc_indices = list()
            n_vertices = len(vertex_dict)
            for i in range(n_vertices):
                this_index = np.where(stc.vertices[0] == \
                                      vertex_dict[i][roi_name])[0][0]
                stc_indices.append(this_index)
            data_to_plot = stc.data[stc_indices[0], :]
            for i in range(1, n_vertices):
                data_to_plot += stc.data[stc_indices[i], :]
            data_to_plot /= n_vertices
           
        legends.append(legend_rewriting(combination['event']))
        plt.plot(stc.times * 1e3, data_to_plot)
        these_data.append(data_to_plot)
        
    plt.ylim(ylim[0], ylim[1])
    plt.legend(legends)
    plt.xlabel('Time (ms)')
    if weight_norm == 'unit-noise-gain-invariant':
        plt.ylabel('Source strength, unit-noise-gain (T)')
    elif weight_norm == 'unit-gain':
        plt.ylabel('Source current density (Am)')
    plt.title(roi_rewriting(roi_name))
    ## stats
    if cluster_times is not None:
        for cluster_time_index, cluster_time in enumerate(cluster_times[1]):
            cluster_index_begin = find_nearest(stc.times, cluster_time[0])
            cluster_index_end   = find_nearest(stc.times, cluster_time[-1])
            print(cluster_index_begin)
            print(cluster_index_end)
            print(cluster_time)
            if 'w' in combination['event']:
                plt.fill_between(cluster_time * 1e3,
                      these_data[0][cluster_index_begin:(cluster_index_end+1)],
                      these_data[1][cluster_index_begin:(cluster_index_end+1)],
                      color=colours[1], alpha=0.5)
            else:
                plt.fill_between(cluster_time * 1e3,
                      these_data[0][cluster_index_begin:(cluster_index_end+1)],
                      these_data[1][cluster_index_begin:(cluster_index_end+1)],
                      alpha=0.5)
    plt.show()
    filename = 'fig2_evoked_stc_' + combination['event'] + '_' + \
        roi_name.replace(' ', '_') + ' ' + weight_norm + '_watershed'
    if save:
        fig.savefig(join(figure_path, filename), dpi=300)
        
        
def get_difference_vertices_evoked(contrast, vertex_dict, roi_name,
                        data_evoked,
                        weight_norm='unit-noise-gain-invariant'):
    ga_path = fname.source_evoked_beamformer_grand_average(
    subject='fsaverage',
    date='20210825_000000',
    fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
    event='s1', first_event='s1',
                second_event='s2',
                weight_norm=weight_norm,
                n_layers=1)
    fsaverage_stc = mne.read_source_estimate(ga_path)
    
    
    
    this_data = data_evoked[contrast]
    n_subjects = len(this_data)
    n_times = this_data[0].shape[1]
    difference_evoked = np.zeros(shape=(n_subjects, n_times))
    for subject_index in range(len(this_data)):
        this_subject_data = this_data[subject_index]
        
        vertex_dict
        if type(vertex_dict) is dict:
            vertex_index = vertex_dict[roi_name]
            stc_index = \
                np.where(fsaverage_stc.vertices[0] == vertex_index)[0][0]
        this_subject_stc_data = this_subject_data[stc_index, :]
        difference_evoked[subject_index, :] = this_subject_stc_data
                
    return difference_evoked
                
            
def plot_difference_vertices_evoked(times, difference, title=None):
    
    mean_difference = np.mean(difference, axis=0)
    SEM_difference = np.std(difference, axis=0) / np.sqrt(len(difference))

    y_extremum = np.max(np.abs(mean_difference))
    y_extremum *= 1.5

    set_rc_params()
    fig = plt.figure()
    plt.plot(times, mean_difference)
    plt.fill_between(times, mean_difference + SEM_difference,
                     mean_difference - SEM_difference,
                     alpha=0.4)
    plt.xlim(-200, 400)
    plt.ylim(-y_extremum, y_extremum)
    plt.hlines(0, times[0], times[-1], color='k', linestyles='dashed')
    plt.vlines(0, -y_extremum, y_extremum, color='k', linestyles='dashed')
    plt.xlabel('Time (ms)')
    plt.ylabel('Difference in source strength (T)')
    if title is not None:
        plt.title(title)
    plt.show()
    
    
    return fig
        
        
def plot_whole_brain_evoked(subject, date, combination, weight_norm,
                            n_layers, max_vertices_dict, time, roi,
                            contrast, src):

    # set_rc_params(font_size=16  )


    def save_T1_plot_only(fig, filename, time, weight_norm, settings):
        weight_norm = weight_norm.replace('-', '_')
        # set_rc_params(font_size=32)

        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.showMaximized()
        axis_0 = fig.axes[0]
        extent = \
            axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        cbar_text = axis_0.get_children()[-2]
        # cbar_text.set_position((0.0, 0-0))

        cbar_text.set_text(settings[weight_norm] + str(time * 1e3) + ' ms')
        cbar_text.set_fontsize('medium')
        fig.savefig(filename, dpi=300, bbox_inches=extent.expanded(1.00, 1.00))

    settings = dict(
        unit_gain='First Stimulation: Source strength, unit gain (Am) at: ',
        unit_noise_gain_invariant='First Stimulation: Source strength,' +  \
                                ' unit-noise-gain (T) at: ')


    ga_path = fname.source_evoked_beamformer_grand_average(
        subject=subject,
        date=date,
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm=weight_norm,
                    n_layers=1)
        
    stc = mne.read_source_estimate(ga_path)
    stc._data = np.abs(stc.data)
    
    # this_stc.crop(tmax=0.060)
    fig = stc.plot(src, initial_time=time,
              clim=dict(kind='value', lims=(np.quantile(stc.data, 0.98),
                                            np.quantile(stc.data, 0.99),
                                            np.quantile(stc.data, 0.999))),
              initial_pos=src[0]['rr'][max_vertices_dict[contrast][0][roi], :])
    fig.set_size_inches(13, 9.75)
    
    save_T1_plot_only_hilbert(fig, join(figure_path, 'fig2_' + contrast[:2] + '_' + \
                                roi +'_T1_' + weight_norm), time)
        
#%% FIG 3

def get_max_vertices(subject, date,
                     event, first_event, second_event, label_dict, src):


    ga_path_1 = fname.source_evoked_beamformer_grand_average(
        subject=subject,
        date=date,
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=event, first_event=first_event, second_event=second_event,
        weight_norm='unit-noise-gain-invariant',
        n_layers=1)
    print('Reading: ' + ga_path_1)
    stc_1 = mne.read_source_estimate(ga_path_1) 
    ga_path_2 = fname.source_evoked_beamformer_grand_average(
        subject=subject,
        date=date,
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=second_event, first_event=first_event, second_event=second_event,
        weight_norm='unit-noise-gain-invariant',
        n_layers=1)
    print('Reading: ' + ga_path_2)
    stc_2 = mne.read_source_estimate(ga_path_2)
    
    # NEW ATTEMPT - averages the two max time courses
    
    max_vertices_1 = find_max_vertex(stc_1, label_dict, 
                                  stc_full_path=ga_path_1, src=src)
    
    max_vertices_2 = find_max_vertex(stc_2, label_dict, 
                                  stc_full_path=ga_path_2, src=src)
    
    return max_vertices_1, max_vertices_2

def create_comparison(roi, weight_norm, diff, fmin=None, fmax=None):
    comparison = dict()
    if 'w' in diff:
        first_event = 'w0'
        second_event = 'w15'
    if 'o' in diff:
        first_event = 'o0'
        second_event = 'o15'
    comparison['diff'] = diff
    comparison['first_event'] = first_event
    comparison['second_event'] = second_event
    comparison['weight_norm'] = weight_norm
    comparison['vertex_index'] = roi['vertex_index']
    comparison['time_index'] = roi['time_index']
    if fmin is not None:
        comparison['fmin'] = fmin
    if fmax is not None:
        comparison['fmax'] = fmax    
    
    return comparison


def get_peak_differences(recordings, bad_subjects, comparison, roi):

    diffs = np.zeros(shape=(len(recordings)))
    for recording_index, recording in enumerate(recordings):
        subject = recording['subject']
        print('Loading subject: ' + subject)
        date = recording['date']
        if subject in bad_subjects:
            diffs[recording_index] = np.nan
            continue
        
        stc_filename_1 = \
            fname.source_evoked_beamformer_morph(
                subject=subject, date=date, fmin=None, fmax=40,
                tmin=-0.200, tmax=1.000, reg=0.00,
                event=comparison['first_event'],
                first_event=comparison['first_event'], 
                second_event=comparison['second_event'],
                weight_norm=comparison['weight_norm'], n_layers=1)
        stc_filename_2 = \
            fname.source_evoked_beamformer_morph(
                subject=subject, date=date, fmin=None, fmax=40,
                tmin=-0.200, tmax=1.000, reg=0.00,
                event=comparison['second_event'],
                first_event=comparison['first_event'], 
                second_event=comparison['second_event'],
                weight_norm=comparison['weight_norm'], n_layers=1)
            
        stc_1 = mne.read_source_estimate(stc_filename_1)
        stc_1._data = abs(stc_1.data)
        stc_2 = mne.read_source_estimate(stc_filename_2)
        stc_2._data = abs(stc_2.data)
        
        
        ## ---------------
        if type(comparison['vertex_index']) is list:
            this_vertex_index_1 = \
        np.where(stc_1.vertices[0] == comparison['vertex_index'][0][roi])[0][0]
            this_vertex_index_2 = \
        np.where(stc_2.vertices[0] == comparison['vertex_index'][1][roi])[0][0]

    
            this_data_1 = stc_1.data[this_vertex_index_1,
                                     comparison['time_index']] 
            this_data_1 += stc_1.data[this_vertex_index_2,
                                      comparison['time_index']]
            this_data_1 /= 2 
                        
            
            this_data_2 = stc_2.data[this_vertex_index_1,
                                     comparison['time_index']]
            this_data_2 += stc_2.data[this_vertex_index_2,
                                      comparison['time_index']]
            this_data_2 /= 2
        ### -------------
        elif type(comparison['vertex_index']) is int:
            this_data_1 = stc_1.data[comparison['vertex_index'],
                     comparison['time_index']]
            this_data_2 = stc_2.data[comparison['vertex_index'],
                     comparison['time_index']]
        
        diffs[recording_index] = np.mean(this_data_1) - np.mean(this_data_2)
    return diffs



def read_subject_slopes_csv(bad_subjects):
    
    path = fname.behavioural_path + '/subject_slopes.csv'
    import csv
    with open(path, 'r') as csvfile:
        csv_file = csv.reader(csvfile)
        weak = list()
        omission = list()
        for line_index, line in enumerate(csv_file):
            if line_index == 0: ## header
                continue
            subject = line[2]
            if subject in bad_subjects:
                weak.append(np.nan)
                omission.append(np.nan)
                continue
            weak.append(float(line[0]))
            omission.append(float(line[1]))

    return weak, omission

def read_glmer_coefficients():
    
    path = fname.behavioural_path + '/full_model_coefficients.csv'
    import csv
    with open(path, 'r') as csvfile:
        csv_file = csv.reader(csvfile)
        for line_index, line in enumerate(csv_file):
            if line_index == 0: # header
                print(line)
            else:
                intercept    = float(line[0])
                stim_type    = float(line[1])
                slope        = float(line[2])
                slope_change = float(line[3])
    return intercept, stim_type, slope, slope_change

def inv_logit(x):
    x_inv = np.exp(x) / (1 + np.exp(x))
    return x_inv
            

#%% FIGs 4 & 5

def load_data_hilbert(combination, recordings, bad_subjects,
                      n_layers, fmin, fmax):
    for recording_index, recording in enumerate(recordings):
        subject = recording['subject']
        date = recording['date']
        if subject in bad_subjects:
            continue
        print(subject)
        full_path = fname.source_hilbert_beamformer_morph(
                subject=subject,
                date=date,
                fmin=fmin, fmax=fmax,
                tmin=-0.750, tmax=0.750,
                event=combination['contrast'][0]['event'],
                first_event=combination['contrast'][0]['first_event'],
                second_event=combination['contrast'][0]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=n_layers)
        full_path2 = fname.source_hilbert_beamformer_morph(
                subject=subject,
                date=date,
                fmin=fmin, fmax=fmax,
                tmin=-0.750, tmax=0.750,
                event=combination['contrast'][1]['event'],
                first_event=combination['contrast'][1]['first_event'],
                second_event=combination['contrast'][1]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=n_layers)
        
        stc = mne.read_source_estimate(full_path)
        stc2 = mne.read_source_estimate(full_path2)
        temp = np.expand_dims(stc.data.copy(), 0)
        temp2 = np.expand_dims(stc2.data.copy(), 0)
        
        ratio = np.array((temp - temp2) / (temp + temp2))
        ratio[np.isnan(ratio)] = 0
        
        if recording_index == 0:
            data = ratio
        else:
            data = np.concatenate((data, ratio), axis=0)
            
    return data


## plot full thing

def plot_full_cluster(comparison, fmin, fmax, stat_tmin, stat_tmax,
                      roi=None, stc_indices=None,
                      src_indices=None, src=None,
                      pos=None, time=None, mode='stat_map',
                      lims=None):
    set_rc_params(font_size=8)
    events = comparison.split('_')

    stc_full_path_1 = \
            fname.source_hilbert_beamformer_grand_average(
                subject='fsaverage', date='20210825_000000', fmin=fmin,
                fmax=fmax,
                tmin=-0.750, tmax=0.750, reg=0.00,
                event=events[0], first_event=events[0], second_event=events[1],
                weight_norm='unit-noise-gain-invariant', n_layers=1)
            
    stc_1 = mne.read_source_estimate(stc_full_path_1)
    
    stc_full_path_2 = \
            fname.source_hilbert_beamformer_grand_average(
                subject='fsaverage', date='20210825_000000', fmin=fmin,
                fmax=fmax,
                tmin=-0.750, tmax=0.750, reg=0.00,
                event=events[1], first_event=events[0], second_event=events[1],
                weight_norm='unit-noise-gain-invariant', n_layers=1)
            
    stc_2 = mne.read_source_estimate(stc_full_path_2)

    ratio = stc_1.copy()
    ratio._data = (stc_1.data - stc_2.data) / (stc_1.data + stc_2.data)
    ratio.crop(stat_tmin, stat_tmax)
    
    stats_filename = fname.source_hilbert_beamformer_statistics(
        subject='fsaverage', date='20210825_000000', fmin=fmin, fmax=fmax,
        tmin=-0.750, tmax=0.750, reg=0.00, first_event=events[0],
        second_event=events[1], stat_tmin=stat_tmin, stat_tmax=stat_tmax,
        nperm=1024, seed=7, condist=None, pval=0.05,
        weight_norm='unit-noise-gain-invariant', n_layers=1)

    stats = np.load(stats_filename, allow_pickle=True).item()
    print('p = ' + str(np.min(stats['cluster_p_values'])))
    

    if np.min(stats['cluster_p_values']) < 0.05:

        c_time_indices = \
          stats['clusters'][np.where(stats['cluster_p_values'] < 0.05)[0][0]][0]
        c_stc_indices  = \
          stats['clusters'][np.where(stats['cluster_p_values'] < 0.05)[0][0]][1]
        
        inv_mask = np.ones(ratio.data.shape, dtype=int)
        for indices in zip(c_stc_indices, c_time_indices):
            inv_mask[indices[0], indices[1]] = 0
            
        ratio._data[inv_mask.astype(bool)] = 0
    else:
        ratio._data[:, :] = 0
    if roi is not None:
            src_index = np.argmax(np.max(ratio.data[stc_indices[roi], :], 
                                          axis=1))
            time_index = np.argmax(np.max(ratio.data[stc_indices[roi], :], 
                                          axis=0))
            if time is None:
                time = ratio.times[time_index]
            print(src_index)
            pos = src[0]['rr'][src_indices[roi][src_index], :]
    if lims is None:
        lims = (0.005, 0.02, 0.03)
    fig = ratio.plot(src, initial_pos=pos, initial_time=time,
                      clim=dict(kind='value', lims=lims),
                      mode=mode)
    return fig, stats, time


# def save_T1_plot_only_hilbert(fig, T1_filename, time):
#     set_rc_params(font_size=8)
#     fig_manager = plt.get_current_fig_manager()
#     fig_manager.window.showMaximized()
#     axis_0 = fig.axes[0]
#     extent = \
#         axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     if time is not None:
#         cbar_text = axis_0.get_children()[-2]
#         cbar_text.set_position((1.0, 1.5))
#         cbar_text.set_text('Power change: (x$_1$ - x$_2$) / (x$_1$ + x$_2$) ' + \
#                             str(round(time * 1e3, 1)) + ' ms')
#         cbar_text.set_fontsize('medium')
        
#         # cbar_text.set_color('white')
#         # cbar_text.set_x(0.5)
#         # cbar_text.set_y(1.1)
#     print(fig.axes[0].get_children())
#     fig.savefig(T1_filename, dpi=300, bbox_inches=extent.expanded(1.00, 1.00))
#     # print(fig.axes[0].get_children()[-2])
    
def get_cluster_times_label(stats, stc_indices, roi, alpha=0.05):
    clusters = stats['clusters']
    cluster_pv = stats['cluster_p_values']
    
    cluster_times = list()
    for cluster_index, cluster in enumerate(clusters):
        if cluster_pv[cluster_index] >= alpha:
            continue
        cluster_time_indices = cluster[0]
        cluster_stc_indices  = cluster[1]
        check_counter = 0
        for counter, cluster_stc_index in enumerate(cluster_stc_indices):
            if cluster_stc_index in stc_indices[roi]:
                check_counter += 1
                cluster_times.append(cluster_time_indices[counter])
                
    cluster_times = np.array(cluster_times)
    cluster_times = np.unique(cluster_times)
    cluster_times = cluster_times / 1e3
    
    ## separate clusters
    these_clusters = list()
    next_cluster_begin_index = 0
    for time_index in range(len(cluster_times) - 1):
        t1 = cluster_times[time_index]
        t2 = cluster_times[time_index + 1]
        ## checking whether it is greater than 0.001 s or at the end
        if (t2 - t1) > 0.0015 or (time_index + 2) == len(cluster_times): 
            this_cluster_end_index  = time_index 
            # print(time_index)
            if (t2 - t1) > 0.0015:
                this_slice = slice(next_cluster_begin_index,
                                   this_cluster_end_index+1)
            elif (time_index + 2) == len(cluster_times):
                this_slice = slice(next_cluster_begin_index,
                                   this_cluster_end_index+2)

            these_clusters.append(cluster_times[this_slice])
            if (time_index + 2) <= len(cluster_times):
                next_cluster_begin_index = time_index + 1
            
    return these_clusters
    
    
def plot_vertices_hilbert(combinations, vertex_dict, roi_name, ylim,
                      fmin, fmax, fig_prepend,
                      cluster_times=None,  time_ms=None,
                      xlim=None, 
                      save=True,
                      subject='fsaverage',
                      date='20210825_000000'):
    f_text = str(fmin) + '_' + str(fmax) + '_Hz'

    fig = plt.figure()
    fig.set_size_inches(8, 6)
    set_rc_params(font_size=14, font_weight='bold')
    legends = list()
    
    these_data = list()
    
    for combination_index, combination in enumerate(combinations):
        ga_path = fname.source_hilbert_beamformer_grand_average(
        subject=subject,
        date=date,
        fmin=fmin, fmax=fmax, tmin=-0.750, tmax=0.750, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm='unit-noise-gain-invariant',
                    n_layers=1)
        stc = mne.read_source_estimate(ga_path)
        if 's' in combination['event']:
            colours = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                       '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        if 'w' in combination['event']:
            colours = ['#2ca02c', '#d62728', '#9467bd',
                              '#8c564b', '#e377c2', '#7f7f7f',
                              '#bcbd22', '#17becf']

        if 'o' in combination['event']:
            colours = [
                              '#bcbd22', '#17becf']
            
        mpl.rcParams['axes.prop_cycle'] = \
        cycler('color', colours )


        stc_indices = list()
        n_vertices = len(vertex_dict)
        
        for i in range(n_vertices):

            this_index = np.where(stc.vertices[0] == \
                                  vertex_dict[i])[0][0]
            stc_indices.append(this_index)
        data_to_plot = stc.data[stc_indices[0], :]
        for i in range(1, n_vertices):
            data_to_plot += stc.data[stc_indices[i], :]
        data_to_plot /= n_vertices
        
        legends.append(legend_rewriting(combination['event']))
        plt.plot(stc.times * 1e3, data_to_plot)
        these_data.append(data_to_plot)
        
    plt.legend(legends)
    plt.xlabel('Time (ms)')
    plt.ylabel('Source strength, unit-noise-gain (T)')
    if ylim is None:
        ylim = fig.get_axes()[0].get_ylim()
    plt.vlines(0, ylim[0], ylim[1], linestyles='dashed',
               color='k', linewidth=3)
    roi_name = roi_rewriting(roi_name)
    if time_ms is None:
        plt.title(roi_name)
    else:
        plt.title(roi_name + '\nFirst Stimulation peak: ' + str(int(time_ms)) + ' ms')
    if cluster_times is not None:
        if len(cluster_times) > 0:
            for cluster in cluster_times:
           
                cluster_index_begin = find_nearest(stc.times, cluster[0])
                cluster_index_end   = find_nearest(stc.times, cluster[-1])
              
                plt.fill_between(np.unique(cluster) * 1e3,
                      these_data[0][cluster_index_begin:(cluster_index_end+1)],
                      these_data[1][cluster_index_begin:(cluster_index_end+1)],
                        color=colours[0], alpha=0.5)
    ## stats
   
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()
    filename = fig_prepend + '_hilbert_stc_' + combination['event'] + '_' +  \
        f_text + '_' + roi_name.replace(' ', '_')
    if save:
        fig.savefig(join(figure_path, filename), dpi=300)
        
        
def get_difference_vertices_hilbert(vertex_dict,
                        data_hilbert,
                        weight_norm='unit-noise-gain-invariant'):
    ga_path = fname.source_hilbert_beamformer_grand_average(
    subject='fsaverage',
    date='20210825_000000',
    fmin=14, fmax=30, tmin=-0.750, tmax=0.750, reg=0.00,
    event='s1', first_event='s1',
                second_event='s2',
                weight_norm=weight_norm,
                n_layers=1)
    fsaverage_stc = mne.read_source_estimate(ga_path)
    

    n_subjects = data_hilbert.shape[0]
    n_times = data_hilbert.shape[2]
    difference_hilbert = np.zeros(shape=(n_subjects, n_times))
    for subject_index in range(n_subjects):
        this_subject_data = data_hilbert[subject_index]
        
        stc_indices = list()
        n_vertices = len(vertex_dict)
        
        for i in range(n_vertices):

            this_index = np.where(fsaverage_stc.vertices[0] == \
                                  vertex_dict[i])[0][0]
            stc_indices.append(this_index)
        data_to_plot = this_subject_data[stc_indices[0], :]
        for i in range(1, n_vertices):
            data_to_plot += this_subject_data[stc_indices[i], :]
        data_to_plot /= n_vertices
        
        difference_hilbert[subject_index, :] = data_to_plot
                
    return difference_hilbert
                
            
def plot_difference_vertices_hilbert(times, difference, title=None):
    
    mean_difference = np.mean(difference, axis=0)
    SEM_difference = np.std(difference, axis=0) / np.sqrt(len(difference))

    # y_extremum = np.max(np.abs(mean_difference))
    # y_extremum *= 1.5
    y_extremum = 0.03

    set_rc_params()
    fig = plt.figure()
    plt.plot(times, mean_difference)
    plt.fill_between(times, mean_difference + SEM_difference,
                     mean_difference - SEM_difference,
                     alpha=0.4)
    plt.xlim(-750, 750)
    plt.ylim(-y_extremum, y_extremum)
    # plt.ylim(-0.03, 0.03)
    plt.hlines(0, times[0], times[-1], color='k', linestyles='dashed')
    plt.vlines(0, -y_extremum, y_extremum, color='k', linestyles='dashed')
    plt.xlabel('Time (ms)')
    plt.ylabel('Ratio ($\mathregular{x_1}$-$\mathregular{x_2}$)/' +\
               '($\mathregular{x_1}$+$\mathregular{x_2}$)')
    if title is not None:
        plt.title(title)
    plt.show()
    # $\mathregular{N_i}$
    
    return fig        
        
      
        
#%% FIG 6

def get_data_array(recordings, bad_subjects, fmin, fmax, tmin, tmax, event,
                   data):
    
    print('Loading for range: ' + str(fmin) + '-' + str(fmax) + \
          ' Hz for event: ' + event)
    indices = list()
    for recording_index, recording in enumerate(recordings[:]):
        subject = recording['subject']
        if subject in bad_subjects:
            continue
        date    = recording['date']
        first_event = event[0] + '0'
        second_event = event[0] + '15'
        
        filename = fname.envelope_correlation_morph_data(
                   subject=subject, date=date, fmin=fmin, fmax=fmax,
                   tmin=tmin, tmax=tmax, reg=0.00,
                   weight_norm='unit-noise-gain-invariant',
                   event=event, n_layers=1,
                   first_event=first_event, second_event=second_event)
        
        print('Loading subject: ' + subject)
        this_data = np.load(filename)
        data[recording_index, :, :] = this_data
        indices.append(recording_index)

        
    return data, indices   

    
def get_t(a1, a2, seed, vertex_indices):
    
    n = a1.shape[0]
    ts = list()
    values = dict()
     
    for label in vertex_indices:
        this_diff = \
        a1[:, vertex_indices[seed], :][:, :, vertex_indices[label]] - \
        a2[:, vertex_indices[seed], :][:, :, vertex_indices[label]]
        
        mu = np.nanmean(np.mean(np.median(this_diff, axis=2), axis=1))
        sem = np.nanstd(np.mean(np.median(this_diff, axis=2), axis=1)) / \
            np.sqrt(n)
        
        # print(mu / sem
        ts.append(mu/sem)
        values[label] = dict()
        values[label]['no-jitter'] = \
np.mean(np.median(a1[:, vertex_indices[seed], :][:, :, vertex_indices[label]],
                                        axis=2), axis=1)
        values[label]['jitter'] = \
np.mean(np.median(a2[:, vertex_indices[seed], :][:, :, vertex_indices[label]],
                                        axis=2), axis=1)
    
    ts = np.array(ts)
    ts_dict = dict()
    for label_index, label in enumerate(vertex_indices):
        ts_dict[label] = ts[label_index]
 
    return ts_dict, values


def get_labelled_data(data, roi, vertex_indices, src, mode='median'):
    data =  np.mean(data[:, vertex_indices[roi]], axis=1)
    stc = mne.VolSourceEstimate(data, [src[0]['vertno']], 0, 1)

    n_sources = data.shape[0]
    mask = np.zeros(n_sources)
    for roi in vertex_indices:
        mask[vertex_indices[roi]] = 1

    mask = mask.astype(bool)

    stc._data[~mask, :] = 0
    
    if mode == 'median':
        for roi in vertex_indices:
            stc._data[vertex_indices[roi]] = \
                np.nanmedian(stc.data[vertex_indices[roi]])
        
    return stc


def get_median_stc(a1, a2, seed, vertex_indices, src, ts_dict=None):

    n_subjects = len(a1)
    n_sources = a1.shape[1]
    subject_median_diff = np.zeros(shape=(n_subjects, n_sources))
    cutoff = abs(t.ppf(0.025, n_subjects-2))
    
    
    for subject_index in range(n_subjects):
        # testing
        # print(subject_index)
        temp = get_labelled_data(a1[subject_index, :, :], seed, 
                                 vertex_indices, src)
        
        
        temp2 = get_labelled_data(a2[subject_index, :, :], seed,
                                  vertex_indices, src)
        
        diff = temp.copy()
        diff._data -= temp2.data
        
        subject_median_diff[subject_index, :] = diff.data.squeeze()
        
    this_data = np.nanmean(subject_median_diff, axis=0)
    if ts_dict is not None:
        for source_index in range(n_sources):
            for label in vertex_indices:
                if source_index in vertex_indices[label]:
                    these_ts = ts_dict
                    this_t = these_ts[label]
                    if this_t > -cutoff and this_t < cutoff:
                        this_data[source_index] = 0
                    
        
    median_stc = mne.VolSourceEstimate(this_data, [src[0]['vertno']], 0, 1)
    
    return median_stc

def save_T1_plot_only_envelope(fig, filename):
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    axis_0 = fig.axes[0]
    extent = \
        axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=300, bbox_inches=extent.expanded(1.05, 1.17))
    
    
def do_anova(values_w, values_o, roi, filename, plot=True, handle_nans=False,
             save=True):
    this_w_no_jitter = values_w[roi]['no-jitter']
    this_w_jitter    = values_w[roi]['jitter']
    this_o_no_jitter = values_o[roi]['no-jitter']
    this_o_jitter    = values_o[roi]['jitter']
    
    dv = np.concatenate((this_w_no_jitter, this_w_jitter,
                         this_o_no_jitter, this_o_jitter))
    iv_stim   = np.repeat(['weak', 'omission'], len(dv) / 2)
    iv_jitter = np.tile(np.repeat(['no-jitter', 'jitter'], len(dv) / 4), 2)
    
    dt = pd.DataFrame(
                      {
                          'envelope_correlation': dv,
                          'stimulation': iv_stim,
                          'regularity': iv_jitter
                          }
                      )
    
    model = ols('envelope_correlation ~ C(stimulation) + C(regularity) + ' + \
                'C(stimulation):C(regularity)', data=dt)

    model = model.fit()
    print(sm.stats.anova_lm(model, typ=2))
    print(model.summary())
    
    if plot:
        set_rc_params()
        means = np.array([np.mean(this_w_no_jitter), np.mean(this_w_jitter),
                 np.mean(this_o_no_jitter), np.mean(this_o_jitter)])
        
        sems = np.array(
            [np.std(this_w_no_jitter)  / np.sqrt(len(this_w_no_jitter)),
                np.std(this_w_jitter)  / np.sqrt(len(this_w_jitter)),
                np.std(this_o_no_jitter)  / np.sqrt(len(this_o_no_jitter)),
                np.std(this_o_jitter)  / np.sqrt(len(this_o_jitter))]
            )
        
        if handle_nans:
            set_rc_params()

            means = np.array([np.nanmean(this_w_no_jitter),
                              np.nanmean(this_w_jitter),
                     np.nanmean(this_o_no_jitter), np.nanmean(this_o_jitter)])
            
            sems = np.array(
            [np.nanstd(this_w_no_jitter)  / \
             np.sqrt(len(this_w_no_jitter[~np.isnan(this_w_no_jitter)])),
            np.nanstd(this_w_jitter)  / \
                np.sqrt(len(this_w_jitter[~np.isnan(this_w_no_jitter)])),
            np.nanstd(this_o_no_jitter)  / \
                np.sqrt(len(this_o_no_jitter[~np.isnan(this_w_no_jitter)])),
            np.nanstd(this_o_jitter)  / \
                np.sqrt(len(this_o_jitter[~np.isnan(this_w_no_jitter)]))])
            
                
        fig = plt.figure(figsize=(8, 6))
        plt.plot((0, 1), means[:2],'-', color='#2ca02c')
        if roi == 'Thalamus_R':
            plt.ylim(0.289, 0.298)
        if roi == 'Thalamus_L':
            plt.ylim((0.2875, 0.3025))
        plt.xlabel('Regularity')
        plt.ylabel('Envelope correlation')
        axis = fig.axes[0]
        axis.set_xticks([0, 1])
        axis.set_xticklabels(['No jitter', 'Jitter'])
        y_ticks = axis.get_yticks()
        axis.set_yticks([y_ticks[1], np.mean(y_ticks[4:6]), y_ticks[-2]])
        plt.plot((0, 1), means[2:], '-', color='#bcbd22')
        plt.legend(['Weak', 'Omission'])

        plt.errorbar(0, means[0], sems[0], ecolor='#2ca02c', capsize=5)
        plt.errorbar(1, means[1], sems[1], ecolor='#2ca02c', capsize=5)
        plt.errorbar(0, means[2], sems[2], ecolor='#bcbd22', capsize=5)
        plt.errorbar(1, means[3], sems[3], ecolor='#bcbd22', capsize=5)
        if roi == 'Thalamus_R':
            plt.title(
                'Interaction plot; cerebello-thalamic (right) correlation')
        elif roi == 'Thalamus_L':
            plt.title(
                'Interaction plot; cerebello-thalamic (left) correlation')
        else:
            plt.title(roi)
        plt.show()
        if save:
            fig.savefig(join(figure_path,
                         'fig6_envelope_freesurfer_ANOVA_' + roi + '.png'),
                        dpi=300)    
            
            
#%% FIG 7

def find_peak(stc, tmin, tmax, vertex):
    this_stc = stc.copy()
    this_stc.crop(tmin, tmax)
    this_data = this_stc.data[stc.vertices[0] == vertex, :]
    this_max = np.max(this_data)
    this_argmax = np.argmax(this_data)
    this_max_time = this_stc.times[this_argmax]
    
    return this_max_time, this_max


def plot_peak_time_courses(stc, plot_dict, figure_path, save=False):
    for roi in plot_dict:
        vertex = plot_dict[roi]['vertex']
        pos = plot_dict[roi]['pos']
        fig = plt.figure()
        fig.set_size_inches(8, 6)
        plt.plot(stc.times * 1e3,
                 stc.data[stc.vertices[0] == vertex, :].T)
        ylim = fig.get_axes()[0].get_ylim()
        plt.vlines(0, ylim[0], ylim[1], color='k', linestyles='dashed')
        roi_rewritten = roi_rewriting(roi)
        title = roi_rewritten + ' (14-30 Hz)\nMNI coordinates: ' + \
            str(int(pos[0] * 1e3)) + ' mm, ' + \
            str(int(pos[1] * 1e3)) + ' mm, ' + \
            str(int(pos[2] * 1e3)) + ' mm'
        plt.title(title, fontdict=dict(fontsize=14))
        peak = find_peak(stc, 0.000, 0.200, vertex)     
        plt.text(peak[0] * 1e3 + 25, peak[1],
                 str(np.round(peak[0] * 1e3, 1)) + ' ms',
                     horizontalalignment='left')
        plt.xlabel('Time (ms)')
        plt.ylabel('First stimulation: Source strength, unit-noise-gain (T)')
        plt.show()
        
        if save:
            filename = 'fig7_thalamus_highlight_' + roi_rewritten + '.png'
            print('Saving: ' + filename)
            fig.savefig(join(figure_path, filename), dpi=300)            
            
            
#%% SUPPLEMENTARY FIGURE 1

def read_staircase_coefficients():
    
    path = fname.behavioural_path
    group_coef = 'staircase_group_coefficients.csv'
    subj_coef  = 'staircase_subject_coefficients.csv'
    
    import csv
    with open(join(path, group_coef)) as csvfile:
         csv_file = csv.reader(csvfile)
         for line_index, line in enumerate(csv_file):
             if line_index == 0:
                 pass
             else:
                 group_intercept = float(line[0])
                 group_slope     = float(line[1])

    subject_intercepts = list()                
    subject_slopes     = list()
                 
                
    with open(join(path, subj_coef)) as csvfile:
        csv_file = csv.reader(csvfile)
        for line_index, line in enumerate(csv_file):
            if line_index == 0:
                pass
            else:
                subject_intercepts.append(float(line[0]))
                subject_slopes.append(float(line[1]))
                                      
    return group_intercept, group_slope, subject_intercepts, subject_slopes