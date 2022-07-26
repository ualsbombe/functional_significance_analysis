#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:06:26 2022

@author: lau
"""

from config import fname
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
from cycler import cycler

#%% EVOKED GRAND AVERAGES (FIGURE 2)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
plt.close('all')


subject = 'fsaverage'
date = '20210825_000000'

ga_path = fname.evoked_grand_average_proj_interpolated(
    subject=subject,
    date=date,
    fmin=None, fmax=40, tmin=-0.200, tmax=1.000)

save_path = fname.subject_figure_path(subject=subject, date=date)

evokeds = mne.read_evokeds(ga_path)
for evoked in evokeds:
    evoked.crop(tmax=0.400)

## s1
fig = evokeds[0].plot(picks='mag', titles=dict(mag='First Stimulation'),
                      time_unit='ms', ylim=dict(mag=(-125, 125)), hline=[0])
fig.set_size_inches(8, 6)
fig.savefig(join(save_path, 's1_evoked.png'), dpi=300)
fig = evokeds[0].plot_topomap(ch_type='mag', times=(0.050, 0.124),
                        vmin=-60, vmax=60, time_unit='ms')
fig.savefig(join(save_path, 's1_topo.png'), dpi=300)

## s2
fig = evokeds[1].plot(picks='mag', titles=dict(mag='Second Stimulation'),
                      time_unit='ms', ylim=dict(mag=(-125, 125)), hline=[0])
fig.set_size_inches(8, 6)
fig.savefig(join(save_path, 's2_evoked.png'), dpi=300)

fig = evokeds[1].plot_topomap(ch_type='mag', times=(0.050, 0.124),
                        vmin=-60, vmax=60, time_unit='ms')
fig.savefig(join(save_path, 's2_topo.png'), dpi=300)

## w0
fig = evokeds[-4].plot(picks='mag',
                       titles=dict(mag='Weak Stimulation (No jitter)'),
                      time_unit='ms', ylim=dict(mag=(-125, 125)), hline=[0])
fig.set_size_inches(8, 6)
fig.savefig(join(save_path, 'w0_evoked.png'), dpi=300)
fig = evokeds[-4].plot_topomap(ch_type='mag', times=(0.050, 0.124),
                        vmin=-60, vmax=60, time_unit='ms')
fig.savefig(join(save_path, 'w0_topo.png'), dpi=300)


## w15
fig = evokeds[-3].plot(picks='mag',
                       titles=dict(mag='Weak Stimulation (Jitter)'),
                      time_unit='ms', ylim=dict(mag=(-125, 125)), hline=[0])
fig.set_size_inches(8, 6)
fig.savefig(join(save_path, 'w15_evoked.png'), dpi=300)
fig = evokeds[-3].plot_topomap(ch_type='mag', times=(0.050, 0.124),
                        vmin=-60, vmax=60, time_unit='ms')
fig.savefig(join(save_path, 'w15_topo.png'), dpi=300)

#%% LOAD DATA FOR STATISTICS (EVOKED)

from config import (recordings, bad_subjects,
                    subjects_with_no_BEM_simnibs)

combinations = [
                dict(contrast=[
                    dict(event='s1', first_event='s1', second_event='s2'),
                    dict(event='s2', first_event='s1', second_event='s2'),
                        ]),
                dict(contrast=[
                    dict(event='w0', first_event='w0', second_event='w15'),
                    dict(event='w15', first_event='w0', second_event='w15'),
                        ]),
                dict(contrast=[
                    dict(event='o0', first_event='o0', second_event='o15'),
                    dict(event='o15', first_event='o0', second_event='o15'),
                        ]),
                ]

def load_data_stat(combination, recordings, bad_subjects,
                   n_layers, excluded_subjects):
    for recording_index, recording in enumerate(recordings):
        subject = recording['subject']
        date = recording['date']
        if subject in bad_subjects or \
            (subject in excluded_subjects and \
             n_layers == 3):
            continue
        print(subject)
        full_path = fname.source_evoked_beamformer_simnibs_morph(
                subject=subject,
                date=date,
                fmin=None, fmax=40,
                tmin=-0.200, tmax=1.000,
                event=combination['contrast'][0]['event'],
                first_event=combination['contrast'][0]['first_event'],
                second_event=combination['contrast'][0]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=n_layers)
        full_path2 = fname.source_evoked_beamformer_simnibs_morph(
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

data = dict()

data['s1_s2'] = load_data_stat(combinations[0], recordings, bad_subjects, 3,
                               subjects_with_no_BEM_simnibs)
data['w0_w15'] = load_data_stat(combinations[1], recordings, bad_subjects, 3,
                                 subjects_with_no_BEM_simnibs)
data['o0_o15'] = load_data_stat(combinations[2], recordings, bad_subjects, 3,
                                subjects_with_no_BEM_simnibs)

#%% STATS

from nilearn import datasets, image

## https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
def find_nearest(array, value):
    array = np.asarray(array)
    index = (np.abs(array - value)).argmin()
    return index

def get_clusters(data, tmin, tmax, stc, vertex_index, alpha):
    tmin_index = find_nearest(stc.times, tmin)
    tmax_index = find_nearest(stc.times, tmax)
    stc_index = np.where(stc.vertices[0]  == vertex_index)[0][0]
    test_data = data[:, stc_index, tmin_index:tmax_index]
    test_times = stc.times[tmin_index:tmax_index]
    
    t_obs, clusters, cluster_pv, H0 = \
        mne.stats.permutation_cluster_1samp_test(test_data, n_permutations=1e3,
                                                 seed=7)
    
    cluster_indices = np.where(cluster_pv < alpha)[0]
    print(cluster_pv)
    cluster_times = list()
    for cluster_index in cluster_indices:
        cluster_time = test_times[clusters[cluster_index][0]]
        cluster_times.append(cluster_time)
        
    return cluster_times

def find_max_vertex(stc, label_dict, stc_full_path, src):
    vertices = src[0]['vertno']
    nifti_full_path = stc_full_path[:-2] + 'nii'
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
    return max_vertices
                   

label_dict = dict(
                SI= dict(label='Postcentral_L', atlas='AAL',
                          restrict_time_index=261),
                SII=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=None),
                CL6=dict(label='Cerebelum_6_L', atlas='AAL',
                         restrict_time_index=None),
                PT=dict(label='Putamen_L', atlas='AAL',
                          restrict_time_index=None),
                PI=dict(label='Parietal_Inf_L', atlas='AAL',
                          restrict_time_index=None),
                # TH=dict(label='Thalamus_L', atlas='AAL',
                #           restrict_time_index=None),

                 )

mr_path = fname.subject_bem_path(subject='fsaverage')



stats = dict(
                s1_s2=dict(),
                w0_w15=dict(),
                o0_o15=dict()
              )
max_vertices_dict = dict()
src = mne.read_source_spaces(join(mr_path, 'volume-7.5_mm-src.fif'))

for comparison_index, comparison in enumerate(stats):
    event = combinations[comparison_index]['contrast'][0]['event']
    first_event = combinations[comparison_index]['contrast'][0]['first_event']
    second_event = combinations[comparison_index]['contrast'][0]['second_event']

    ## find mean - don't bias
    ga_path = fname.source_evoked_beamformer_grand_average_simnibs(
        subject='fsaverage',
        date='20210825_000000',
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=event, first_event=first_event, second_event=second_event,
        weight_norm='unit-noise-gain-invariant',
        n_layers=3)
    stc_1 = mne.read_source_estimate(ga_path) 
    ga_path = fname.source_evoked_beamformer_grand_average_simnibs(
        subject='fsaverage',
        date='20210825_000000',
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=second_event, first_event=first_event, second_event=second_event,
        weight_norm='unit-noise-gain-invariant',
        n_layers=3)
    stc_2 = mne.read_source_estimate(ga_path)
    
    stc = stc_2.copy()
    stc._data = abs(stc.data)
    # stc._data -= abs(stc_2.data)
    # stc._data /
    
    max_vertices = find_max_vertex(stc, label_dict, 
                                  stc_full_path=ga_path, src=src)
    max_vertices_dict[comparison] = max_vertices
    for max_vertex in max_vertices:
        print('\n' + comparison + ' ' + max_vertex +  '\n')
        stats[comparison][max_vertex] = \
            get_clusters(data[comparison], 0.000, 0.150, stc,
                          max_vertices[max_vertex],
                          alpha=0.05)

#%% STCS EVOKED TIME COURSES

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
plt.close('all')


def legend_rewriting(legend):
    if legend == 's1':
        legend = 'First Stimulation'
    elif legend == 's2':
        legend = 'Second Stimulation'
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

def find_peaks(stc, tmins, tmaxs, vertex_index):
    assert(len(tmins) == len(tmaxs))
    times = list()
    peaks = list()
    for (tmin, tmax) in zip(tmins, tmaxs):
        copy = stc.copy()
        copy.crop(tmin, tmax)
        copy._data = np.abs(copy.data) ## taking abs
        copy_index = np.where(copy.vertices[0]  == vertex_index)[0][0]
        this_data = copy.data.copy()
        this_data = this_data[copy_index, :]
        time_index = np.argmax(this_data)
        peak = np.max(this_data)
        time = copy.times[time_index]
        time *= 1e3 ## make ms
        times.append(time)
        peaks.append(peak)
        
    return times, peaks
        
        
def set_texts(time_peaks, peaks):
    assert(len(time_peaks) == len(peaks))
    for time_peak_index, time_peak in enumerate(time_peaks):
        if len(time_peaks) == 1:
            movement = -40
        if len(time_peaks ) == 2:
            if time_peak_index % 2: # odd or even
                movement = 10
            else:
                movement = -80
        elif len(time_peaks) == 3:
            if time_peak_index == 0 or time_peak_index == 1:
                movement = -80
            elif time_peak_index == 2:
                movement = 10
                
        plt.text(time_peak + movement, peaks[time_peak_index] + 5e-15,
                 str(int(round(time_peak, 1))) + ' ms')


def plot_vertices(combinations, vertex_index, roi_name, ylim,
                  tmins, tmaxs, cluster_times=None,
                  save=True,
                  subject='fsaverage',
                  date='20210825_000000'):
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    legends = list()

    save_path = fname.subject_figure_path(subject=subject, date=date)
    
    these_data = list()
    
    for combination_index, combination in enumerate(combinations):
        ga_path = fname.source_evoked_beamformer_grand_average_simnibs(
        subject=subject,
        date=date,
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm='unit-noise-gain-invariant',
                    n_layers=3)
        stc = mne.read_source_estimate(ga_path)
        if 'w' in combination['event']:
            colours = ['#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f',
                             '#bcbd22', '#17becf']
            mpl.rcParams['axes.prop_cycle'] = \
            cycler('color', colours )


        stc.crop(tmin=-0.050, tmax=0.400) ## cropping
        if (combination_index == 0 and combination['event'] == 's1') or \
           (combination_index == 0 and combination['event'] == 'w0') or \
           (combination_index == 0 and combination['event'] == 'o0'):
            time_peaks, peaks = find_peaks(stc, tmins, tmaxs, vertex_index)
        stc._data = np.abs(stc.data) ## taking abs
        stc_index = np.where(stc.vertices[0]  == vertex_index)[0][0]
        legends.append(legend_rewriting(combination['event']))
        plt.plot(stc.times * 1e3, stc.data[stc_index, :])
        these_data.append(stc.data[stc_index, :])
        
    plt.ylim(ylim[0], ylim[1])
    plt.legend(legends)
    plt.vlines(time_peaks, ylim[0], peaks, linestyles='dashed',
               color='k')
    set_texts(time_peaks, peaks)
    plt.xlabel('Time (ms)')
    plt.ylabel('Source strength, unit-noise-gain (T)')
    plt.title(roi_name)
    ## stats
    if cluster_times is not None:
        for cluster_time_index, cluster_time in enumerate(cluster_times):
            cluster_index_begin = find_nearest(stc.times, cluster_time[0])
            cluster_index_end   = find_nearest(stc.times, cluster_time[-1])
            # print(cluster_index_begin)
            # print(cluster_index_end)
            # print(cluster_time)
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
    filename = 'evoked_stc_' + combination['event'] + '_' + \
        roi_name.replace(' ', '_')
    if save:
        fig.savefig(join(save_path, filename), dpi=300)


#% stim
combination = [
                dict(event='s1', first_event='s1', second_event='s2'),
                dict(event='s2', first_event='s1', second_event='s2'),
                ]

plot_vertices(combination, max_vertices_dict['s1_s2']['SI'], 'Primary somatosensory cortex',
              ylim=(0, 65e-15), tmins=[0.030],
                                tmaxs=[0.070],
                                cluster_times=stats['s1_s2']['SI'])
plot_vertices(combination, max_vertices_dict['s1_s2']['CL6'], 'Cerebellum 6 L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.120, 0.150],
                                cluster_times=stats['s1_s2']['CL6'])
plot_vertices(combination, max_vertices_dict['s1_s2']['SII'], 'Secondary somatosensory cortex',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['SII'])
plot_vertices(combination, max_vertices_dict['s1_s2']['PT'], 'Putamen L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['PT'])
plot_vertices(combination, max_vertices_dict['s1_s2']['PI'], 'Parietal Inf L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['PI'])
# plot_vertices(combination, max_vertices_dict['s1_s2']['TH'], 'Thalamus L',
#               ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
#                                 tmaxs=[0.060, 0.090, 0.130],
#                                 cluster_times=stats['s1_s2']['TH'])


#% weak
combination = [
                dict(event='w0', first_event='w0', second_event='w15'),
                dict(event='w15', first_event='w0', second_event='w15'),
                ]

plot_vertices(combination, max_vertices_dict['w0_w15']['SI'], 'Primary somatosensory cortex',
              ylim=(0, 65e-15), tmins=[0.030],
                                tmaxs=[0.070],
                                cluster_times=stats['w0_w15']['SI'])
plot_vertices(combination, max_vertices_dict['w0_w15']['CL6'], 'Cerebellum 6 L',
              ylim=(0, 65e-15), tmins=[0.100],
                                tmaxs=[0.110],
                                cluster_times=stats['w0_w15']['CL6'])
plot_vertices(combination, max_vertices_dict['w0_w15']['SII'], 'Secondary somatosensory cortex',
              ylim=(0, 65e-15), tmins=[0.090, 0.130],
                                tmaxs=[0.110, 0.150],
                                cluster_times=stats['w0_w15']['SII'])
plot_vertices(combination, max_vertices_dict['w0_w15']['PT'], 'Putamen L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['w0_w15']['PT'])
plot_vertices(combination, max_vertices_dict['w0_w15']['PI'], 'Parietal Inf L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['w0_w15']['PI'])
# plot_vertices(combination, max_vertices_dict['w0_w15']['TH'], 'Thalamus L',
#               ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
#                                 tmaxs=[0.060, 0.090, 0.130],
#                                 cluster_times=stats['w0_w15']['TH'])

#% omission
combination = [
                dict(event='o0', first_event='o0', second_event='o15'),
                dict(event='o15', first_event='o0', second_event='o15'),
                ]

plot_vertices(combination, max_vertices_dict['o0_o15']['SI'], 'Primary somatosensory cortex',
              ylim=(0, 48e-15), tmins=[0.030],
                                tmaxs=[0.070],
                                cluster_times=stats['o0_o15']['SI'])
plot_vertices(combination, max_vertices_dict['o0_o15']['CL6'], 'Cerebellum 6 L',
              ylim=(0, 48e-15), tmins=[0.100],
                                tmaxs=[0.120],
                                cluster_times=stats['o0_o15']['CL6'])
plot_vertices(combination, max_vertices_dict['o0_o15']['SII'], 'Secondary somatosensory cortex',
              ylim=(0, 48e-15), tmins=[0.090, 0.130],
                                tmaxs=[0.110, 0.150],
                                cluster_times=stats['o0_o15']['SII'])
plot_vertices(combination, max_vertices_dict['o0_o15']['PT'], 'Putamen L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['o0_o15']['PT'])
plot_vertices(combination, max_vertices_dict['o0_o15']['PI'], 'Parietal Inf L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['o0_o15']['PI'])
# plot_vertices(combination, max_vertices_dict['o0_o15']['TH'], 'Thalamus L',
#               ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
#                                 tmaxs=[0.060, 0.090, 0.130],
#                                 cluster_times=stats['o0_o15']['TH'])

#%% FULL BRAIN PLOT (EVOKEDS)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.weight'] = 'bold'
plt.close('all')

subject = 'fsaverage'
date = '20210825_000000'

src_path = fname.anatomy_volumetric_source_space(subject=subject, spacing=7.5)
src = mne.read_source_spaces(src_path)
save_path = fname.subject_figure_path(subject=subject, date=date)

def save_T1_plot_only(fig, filename, time, weight_norm, settings):
    mpl.rcParams['font.size'] = 8
    weight_norm = weight_norm.replace('-', '_')

    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    axis_0 = fig.axes[0]
    extent = \
        axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    cbar_text = axis_0.get_children()[-2]
    cbar_text.set_text(settings[weight_norm] + str(time * 1e3) + ' ms')
    cbar_text.set_fontsize('small')
    fig.savefig(filename, dpi=300, bbox_inches=extent.expanded(1.05, 1.17))

combination = dict(event='s1', first_event='s1', second_event='s2')
weight_norms = [
    # 'unit-gain',
    'unit-noise-gain-invariant'
    ]
settings = dict(
    unit_gain='First Stimulation: Source strength, unit gain (Am) at: ',
    unit_noise_gain_invariant='First Stimulation: Source strength,' +  \
                            ' unit-noise-gain (T) at: ')

for weight_norm in weight_norms:

    ga_path = fname.source_evoked_beamformer_grand_average_simnibs(
        subject=subject,
        date=date,
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm=weight_norm,
                    n_layers=3)
    
    stc = mne.read_source_estimate(ga_path)
    stc._data = np.abs(stc.data)
    
    ## SI
    time = 0.051
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2']['SI'], :])
    
    save_T1_plot_only(fig, join(save_path, 's1_SI_T1_' + weight_norm), time,
                      weight_norm, settings)
    
    #SII
    time = 0.123
    fig = stc.plot(src, initial_time=time, 
              clim=dict(kind='value', lims=(np.quantile(stc.data, 0.95),
                                  np.quantile(stc.data, 0.975),
                                  np.quantile(stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2']['SII'], :])
    
    save_T1_plot_only(fig, join(save_path, 's1_SII_T1_' + weight_norm), time,
                      weight_norm, settings)
    
    
    ## save
    
    #Cerebellum 6 L
    time = 0.141
    fig = stc.plot(src, initial_time=time, 
              clim=dict(kind='value', lims=(np.quantile(stc.data, 0.95),
                                  np.quantile(stc.data, 0.975),
                                  np.quantile(stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2']['CL6'], :])
    
    save_T1_plot_only(fig, join(save_path,
                                's1_Cerebellum_6_T1_' + weight_norm), time,
                       weight_norm, settings)
    
    # time = 0.124
    # fig = stc.plot(src, initial_time=time, 
    #           clim=dict(kind='value', lims=(np.quantile(stc.data, 0.99),
    #                               np.quantile(stc.data, 0.995),
    #                               np.quantile(stc.data, 0.999))),
    #           initial_pos=(-0.035, -0.051, -0.032))
    


#%% OSCILLATORY RESPONSES

from config import (recordings, bad_subjects,
                    subjects_with_no_BEM_simnibs)

combinations = [
                dict(contrast=[
                    dict(event='s1', first_event='s1', second_event='s2'),
                    dict(event='s2', first_event='s1', second_event='s2'),
                        ]),
                # dict(contrast=[
                #     dict(event='w0', first_event='w0', second_event='w15'),
                #     dict(event='w15', first_event='w0', second_event='w15'),
                #         ]),
                # dict(contrast=[
                #     dict(event='o0', first_event='o0', second_event='o15'),
                #     dict(event='o15', first_event='o0', second_event='o15'),
                #         ]),
                ]

def load_data_stat(combination, recordings, bad_subjects,
                   n_layers, excluded_subjects):
    for recording_index, recording in enumerate(recordings):
        subject = recording['subject']
        date = recording['date']
        if subject in bad_subjects or subject in excluded_subjects:
            continue
        print(subject)
        full_path = fname.source_hilbert_beamformer_simnibs_morph(
                subject=subject,
                date=date,
                fmin=4, fmax=7,
                tmin=-0.750, tmax=0.750,
                event=combination['contrast'][0]['event'],
                first_event=combination['contrast'][0]['first_event'],
                second_event=combination['contrast'][0]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=n_layers)
        full_path2 = fname.source_hilbert_beamformer_simnibs_morph(
                subject=subject,
                date=date,
                fmin=4, fmax=7,
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




data = dict()

data['s1_s2'] = load_data_stat(combinations[0], recordings, bad_subjects, 3,
                               subjects_with_no_BEM_simnibs)
# data['w0_w15'] = load_data_stat(combinations[1], recordings, bad_subjects, 3,
#                                  subjects_with_no_BEM_simnibs)
# data['o0_o15'] = load_data_stat(combinations[2], recordings, bad_subjects, 3,
#                                 subjects_with_no_BEM_simnibs)

#%% CLUSTERS OSCILLATORY


mr_path = fname.subject_bem_path(subject='fsaverage')

stats = dict(
                s1_s2=dict(),
                # w0_w15=dict(),
                # o0_o15=dict()
              )
max_vertices_dict = dict()
src = mne.read_source_spaces(join(mr_path, 'volume-7.5_mm-src.fif'))
label_dict = dict(
                SI= dict(label='Postcentral_L', atlas='AAL',
                          restrict_time_index=261),
                SII=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=None),
                CL6=dict(label='Cerebelum_6_L', atlas='AAL',
                         restrict_time_index=None),
                PT=dict(label='Putamen_L', atlas='AAL',
                          restrict_time_index=None),
                PI=dict(label='Parietal_Inf_L', atlas='AAL',
                          restrict_time_index=None),
                HI=dict(label='Hippocampus_L', atlas='AAL',
                          restrict_time_index=None),

                 )

for comparison_index, comparison in enumerate(stats):
    event = combinations[comparison_index]['contrast'][0]['event']
    first_event = combinations[comparison_index]['contrast'][0]['first_event']
    second_event = combinations[comparison_index]['contrast'][0]['second_event']

    ## find mean - don't bias
    ga_path = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject='fsaverage',
        date='20210825_000000',
        fmin=4, fmax=7, tmin=-0.750, tmax=0.750, reg=0.00,
        event=event, first_event=first_event, second_event=second_event,
        weight_norm='unit-noise-gain-invariant',
        n_layers=1)
    stc_1 = mne.read_source_estimate(ga_path) 
    ga_path = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject='fsaverage',
        date='20210825_000000',
        fmin=4, fmax=7, tmin=-0.750, tmax=0.750, reg=0.00,
        event=second_event, first_event=first_event, second_event=second_event,
        weight_norm='unit-noise-gain-invariant',
        n_layers=1)
    stc_2 = mne.read_source_estimate(ga_path)
    
    stc = stc_1.copy()
    stc._data = (stc_1.data - stc_2.data) / (stc_1.data + stc_2.data)

    
    max_vertices = find_max_vertex(stc, label_dict, 
                                  stc_full_path=ga_path, src=src)
    max_vertices_dict[comparison] = max_vertices
    for max_vertex in max_vertices:
        print('\n' + comparison + ' ' + max_vertex +  '\n')
        stats[comparison][max_vertex] = \
            get_clusters(data[comparison], -0.750, 0.750, stc,
                          max_vertices[max_vertex],
                          alpha=0.05)



#%% PLOT VERTICES OSCILLATORY

plt.close('all')

def plot_vertices_oscillatory(combinations, vertex_index, roi_name, ylim,
                      tmins, tmaxs, cluster_times=None,
                      save=True,
                      subject='fsaverage',
                      date='20210825_000000'):
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    legends = list()

    save_path = fname.subject_figure_path(subject=subject, date=date)
    
    these_data = list()
    
    for combination_index, combination in enumerate(combinations):
        ga_path = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject=subject,
        date=date,
        fmin=4, fmax=7, tmin=-0.750, tmax=0.750, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm='unit-noise-gain-invariant',
                    n_layers=1)
        stc = mne.read_source_estimate(ga_path)
        if 'w' in combination['event']:
            colours = ['#2ca02c', '#d62728', '#9467bd',
                             '#8c564b', '#e377c2', '#7f7f7f',
                             '#bcbd22', '#17becf']
            mpl.rcParams['axes.prop_cycle'] = \
            cycler('color', colours )


        # stc.crop(tmin=-0.050, tmax=0.400) ## cropping
        if (combination_index == 0 and combination['event'] == 's1') or \
           (combination_index == 0 and combination['event'] == 'w0') or \
           (combination_index == 0 and combination['event'] == 'o0'):
            time_peaks, peaks = find_peaks(stc, tmins, tmaxs, vertex_index)
        # stc._data = np.abs(stc.data) ## taking abs
        stc_index = np.where(stc.vertices[0]  == vertex_index)[0][0]
        legends.append(legend_rewriting(combination['event']))
        plt.plot(stc.times * 1e3, stc.data[stc_index, :])
        these_data.append(stc.data[stc_index, :])
        
    # plt.ylim(ylim[0], ylim[1])
    plt.legend(legends)
    # plt.vlines(time_peaks, ylim[0], peaks, linestyles='dashed',
    #            color='k')
    set_texts(time_peaks, peaks)
    plt.xlabel('Time (ms)')
    plt.ylabel('Source strength, unit-noise-gain (T)')
    plt.title(roi_name)
    ## stats
    if cluster_times is not None:
        for cluster_time_index, cluster_time in enumerate(cluster_times):
            cluster_index_begin = find_nearest(stc.times, cluster_time[0])
            cluster_index_end   = find_nearest(stc.times, cluster_time[-1])
 
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
    filename = 'oscillatory_stc_' + combination['event'] + '_' + \
        roi_name.replace(' ', '_')
    if save:
        fig.savefig(join(save_path, filename), dpi=300)
        
#% stim
combination = [
                dict(event='s1', first_event='s1', second_event='s2'),
                dict(event='s2', first_event='s1', second_event='s2'),
                ]

plot_vertices_oscillatory(combination, max_vertices_dict['s1_s2']['SI'], 'Primary somatosensory cortex',
              ylim=(0, 65e-15), tmins=[0.030],
                                tmaxs=[0.070],
                                cluster_times=stats['s1_s2']['SI'])
plot_vertices_oscillatory(combination, max_vertices_dict['s1_s2']['CL6'], 'Cerebellum 6 L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.120, 0.150],
                                cluster_times=stats['s1_s2']['CL6'])
plot_vertices_oscillatory(combination, max_vertices_dict['s1_s2']['SII'], 'Secondary somatosensory cortex',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['SII'])
plot_vertices_oscillatory(combination, max_vertices_dict['s1_s2']['PT'], 'Putamen L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['PT'])
plot_vertices_oscillatory(combination, max_vertices_dict['s1_s2']['PI'], 'Parietal Inf L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['PI'])
plot_vertices_oscillatory(combination, max_vertices_dict['s1_s2']['HI'], 'Hippocampus L',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['HI'])    