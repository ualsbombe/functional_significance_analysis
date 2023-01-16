#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:06:26 2022

@author: lau
"""

#%% IMPORTS

from config import fname
import mne
import matplotlib as mpl
import matplotlib.pyplot as plt
from os.path import join, exists
import numpy as np
from cycler import cycler
from nilearn import datasets, image

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
#%% EVOKED GRAND AVERAGES (FIGURE 2)

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
# plt.close('all')


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
fig = evokeds[0].plot(picks='mag', titles=dict(mag='Magnetometers: First Stimulation'),
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

#%% EVOKED - THALAMUS BACK-UP

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

mag_mask = create_topo_mask(evokeds[0].copy(), channels, 'mag')
grad_mask = create_topo_mask(evokeds[0].copy(), channels, 'grad')

fig = evokeds[0].plot_topomap(ch_type='mag',
                               times=(0.060, 0.085, 0.124, 0.140, 0.170),
                              # times=0.085,
                                vmin=-60, vmax=60,
                               time_unit='ms', mask=mag_mask,
                               mask_params=dict(markersize=8),
                               show_names=False) 
fig = evokeds[0].plot_topomap(ch_type='grad',
                               times=(0.060, 0.085, 0.124, 0.140, 0.170),
                              # times=0.085,
                                vmin=0, vmax=20,
                               time_unit='ms', mask=grad_mask,
                               mask_params=dict(markersize=8),
                               show_names=False) 
                


mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 12
mpl.rcParams['font.weight'] = 'bold'
plt.close('all')

fig = evokeds[0].plot(picks='mag',
                      titles=dict(mag='Magnetometers:\nFirst Stimulation'),
                      time_unit='ms', ylim=dict(mag=(-125, 125)), hline=[0],
                      xlim=(-50, 200), highlight=(80, 90))
fig.set_size_inches(8, 6)
fig.savefig(join(save_path, 's1_thalamus_highlight_evoked.png'), dpi=300)


fig = evokeds[0].plot_topomap(ch_type='mag',
                               times=(0.060, 0.085, 0.124, 0.140, 0.170),
                                vmin=-60, vmax=60,
                               time_unit='ms', mask=mag_mask,
                               mask_params=dict(markersize=8),
                               show_names=False) 
fig.savefig(join(save_path, 's1_thalamus_highlight_topo.png'), dpi=300)

fig = evokeds[0].plot(picks='grad', 
                  titles=dict(grad='Gradiometers:\nFirst Stimulation'),
                      time_unit='ms', ylim=dict(grad=(-50, 50)), hline=[0],
                      xlim=(-50, 200), highlight=(80, 90))
fig.set_size_inches(8, 6)
fig.savefig(join(save_path, 's1_thalamus_highlight_evoked_grad.png'), dpi=300)
fig = evokeds[0].plot_topomap(ch_type='grad',
                               times=(0.060, 0.085, 0.124, 0.140, 0.170),
                                vmin=0, vmax=20,
                               time_unit='ms', mask=grad_mask,
                               mask_params=dict(markersize=8),
                               show_names=False) 
fig.savefig(join(save_path, 's1_thalamus_highlight_topo_grad.png'), dpi=300)



#%%# stc beta part

full_path = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject='fsaverage',
        date='20210825_000000',
        fmin=14, fmax=30,
        tmin=-0.750, tmax=0.750,
        event='s1',
        first_event='s1',
        second_event='s2',
        reg=0.00, weight_norm='unit-noise-gain-invariant',
        n_layers=3)

# full_path = fname.source_hilbert_beamformer_grand_average_simnibs(
#         subject='fsaverage',
#         date='20210825_000000',
#         fmin=14, fmax=30,
#         tmin=-0.750, tmax=0.750,
#         event='s3',
#         first_event='s2',
#         second_event='s3',
#         reg=0.00, weight_norm='unit-noise-gain-invariant',
#         n_layers=3)

stc = mne.read_source_estimate(full_path)
src_path = fname.anatomy_volumetric_source_space(subject='fsaverage',
                                                 spacing=7.5)
src = mne.read_source_spaces(src_path)
# stc.crop(-0.050, 0.200)
stc.plot(src, 'fsaverage', initial_time=0.060,
         initial_pos=(-0.017, -0.035, 0.071)) ## LSI % 12061
stc.plot(src, 'fsaverage', initial_time=0.107,
         initial_pos=(0.017, -0.035, 0.071)) ## RSI % 12065
stc.plot(src, 'fsaverage', initial_time=0.060,
         initial_pos=(-0.044, -0.032, 0.019)) ## LSII % 8354
stc.plot(src, 'fsaverage', initial_time=0.107,
         initial_pos=(0.044, -0.032, 0.019)) ## RSII % 8366
stc.plot(src, 'fsaverage', initial_time=0.086,
         initial_pos=(-0.010, -0.019, 0.007)) ## L TH % 7140
stc.plot(src, 'fsaverage', initial_time=0.086,
         initial_pos=(0.010, -0.019, 0.007)) ## R TH % 7142


#%% PEAK TIME COURSES - THALAMIC BACKUP

full_path = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject='fsaverage',
        date='20210825_000000',
        fmin=14, fmax=30,
        tmin=-0.750, tmax=0.750,
        event='s1',
        first_event='s1',
        second_event='s2',
        reg=0.00, weight_norm='unit-noise-gain-invariant',
        n_layers=3)

stc = mne.read_source_estimate(full_path)

plt.close('all')

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['lines.linewidth'] = 3

plot_dict = dict(
                    SI=dict(pos=(-0.017, -0.035, 0.071), vertex=12061),
                    SI_R=dict(pos=(0.017, -0.035, 0.071), vertex=12065),
                    SII=dict(pos=(-0.044, -0.032, 0.019), vertex=8354),
                    SII_R=dict(pos=(0.044, -0.032, 0.019), vertex=8366),
                    TH=dict(pos=(-0.010, -0.019, 0.007), vertex=7140),
                    TH_R=dict(pos=(0.010, -0.019, 0.007), vertex=7142)
                )

def find_peak(stc, tmin, tmax, vertex):
    this_stc = stc.copy()
    this_stc.crop(tmin, tmax)
    this_data = this_stc.data[stc.vertices[0] == vertex, :]
    this_max = np.max(this_data)
    this_argmax = np.argmax(this_data)
    this_max_time = this_stc.times[this_argmax]
    
    return this_max_time, this_max


def plot_peak_time_courses(stc, plot_dict, savepath, save=False):
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
            filename = 'thalamus_highlight_' + roi_rewritten + '.png'
            print('Saving: ' + filename)
            fig.savefig(join(savepath, filename), dpi=300)

           

plot_peak_time_courses(stc, plot_dict, save_path, True)


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

data_evoked = dict()

data_evoked['s1_s2'] = load_data_stat(combinations[0], recordings,
                                      bad_subjects, 3,
                               subjects_with_no_BEM_simnibs)
data_evoked['w0_w15'] = load_data_stat(combinations[1], recordings,
                                       bad_subjects, 3,
                                 subjects_with_no_BEM_simnibs)
data_evoked['o0_o15'] = load_data_stat(combinations[2], recordings, 
                                       bad_subjects, 3,
                                subjects_with_no_BEM_simnibs)

#%% STATS

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
                   
#%% RUN STATS EVOKED
label_dict = dict(
                SI= dict(label='Postcentral_L', atlas='AAL',
                          restrict_time_index=261),
                SII=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=351),
                CL6=dict(label='Cerebelum_6_L', atlas='AAL',
                          restrict_time_index=351),
                PT=dict(label='Putamen_L', atlas='AAL',
                          restrict_time_index=351),
                PI=dict(label='Parietal_Inf_L', atlas='AAL',
                          restrict_time_index=351),
                TH=dict(label='Thalamus_L', atlas='AAL',
                          restrict_time_index=351),
                TH_R=dict(label='Thalamus_R', atlas='AAL',
                          restrict_time_index=351),
                PL=dict(label='Pallidum_L', atlas='AAL',
                            restrict_time_index=351)

                 )

mr_path = fname.subject_bem_path(subject='fsaverage')
src = mne.read_source_spaces(join(mr_path, 'volume-7.5_mm-src.fif'))


stats = dict(
                s1_s2=dict(),
                w0_w15=dict(),
                o0_o15=dict()
              )
max_vertices_dict = dict()

for comparison_index, comparison in enumerate(stats):
    event = combinations[comparison_index]['contrast'][0]['event']
    first_event = combinations[comparison_index]['contrast'][0]['first_event']
    second_event = combinations[comparison_index]['contrast'][0]['second_event']

    ## find mean - don't bias
    ga_path_1 = fname.source_evoked_beamformer_grand_average_simnibs(
        subject='fsaverage',
        date='20210825_000000',
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=event, first_event=first_event, second_event=second_event,
        weight_norm='unit-noise-gain-invariant',
        n_layers=3)
    print('Reading: ' + ga_path_1)
    stc_1 = mne.read_source_estimate(ga_path_1) 
    ga_path_2 = fname.source_evoked_beamformer_grand_average_simnibs(
        subject='fsaverage',
        date='20210825_000000',
        fmin=None, fmax=40, tmin=-0.200, tmax=1.000, reg=0.00,
        event=second_event, first_event=first_event, second_event=second_event,
        weight_norm='unit-noise-gain-invariant',
        n_layers=3)
    print('Reading: ' + ga_path_2)
    stc_2 = mne.read_source_estimate(ga_path_2)
    
    
            
    # NEW ATTEMPT - averages the two max time courses
    
    max_vertices_1 = find_max_vertex(stc_1, label_dict, 
                                  stc_full_path=ga_path_1, src=src)
    
    max_vertices_2 = find_max_vertex(stc_2, label_dict, 
                                  stc_full_path=ga_path_2, src=src)
    
    max_vertices_dict[comparison] = [max_vertices_1, max_vertices_2]
    # print(max_vertices_dict)
    
    
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


#%% STCS EVOKED TIME COURSES

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
plt.close('all')



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


def plot_vertices(combinations, vertex_dict, roi_name, ylim,
                  tmins, tmaxs, cluster_times=None,
                  save=True,
                  subject='fsaverage',
                  date='20210825_000000',
                  weight_norm='unit-noise-gain-invariant'):
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
                    weight_norm=weight_norm,
                    n_layers=3)
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
        # if (combination_index == 0 and combination['event'] == 's1') or \
        #    (combination_index == 0 and combination['event'] == 'w0') or \
        #    (combination_index == 0 and combination['event'] == 'o0'):
        #     time_peaks, peaks = find_peaks(stc, tmins, tmaxs, vertex_index)
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
    if weight_norm == 'unit-noise-gain-invariant':
        pass
        #plt.vlines(time_peaks, ylim[0], peaks, linestyles='dashed',
         #          color='k')
        #set_texts(time_peaks, peaks)
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
    filename = 'evoked_stc_' + combination['event'] + '_' + \
        roi_name.replace(' ', '_') + ' ' + weight_norm
    if save:
        fig.savefig(join(save_path, filename), dpi=300)


#%% stim
combination = [
                dict(event='s1', first_event='s1', second_event='s2'),
                dict(event='s2', first_event='s1', second_event='s2'),
                ]

plot_vertices(combination, max_vertices_dict['s1_s2'], 'SI',
              ylim=(0, 65e-15), tmins=[0.030],
                                tmaxs=[0.070],
                                cluster_times=stats['s1_s2']['SI'][1])
plot_vertices(combination, max_vertices_dict['s1_s2'], 'CL6',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.120, 0.150],
                                cluster_times=stats['s1_s2']['CL6'])
plot_vertices(combination, max_vertices_dict['s1_s2'], 'SII',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['SII'])
plot_vertices(combination, max_vertices_dict['s1_s2'], 'PT',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['PT'])
plot_vertices(combination, max_vertices_dict['s1_s2'], 'PI',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['PI'])
plot_vertices(combination, max_vertices_dict['s1_s2'], 'TH',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['TH'])
plot_vertices(combination, max_vertices_dict['s1_s2'], 'PL',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['s1_s2']['PL'])


#%% weak
combination = [
                dict(event='w0', first_event='w0', second_event='w15'),
                dict(event='w15', first_event='w0', second_event='w15'),
                ]

plot_vertices(combination, max_vertices_dict['w0_w15'], 'SI',
              ylim=(0, 65e-15), tmins=[0.030],
                                tmaxs=[0.070],
                                cluster_times=stats['w0_w15']['SI'])
plot_vertices(combination, max_vertices_dict['w0_w15'], 'CL6',
              ylim=(0, 65e-15), tmins=[0.100],
                                tmaxs=[0.110],
                                cluster_times=stats['w0_w15']['CL6'])
plot_vertices(combination, max_vertices_dict['w0_w15'], 'SII',
              ylim=(0, 65e-15), tmins=[0.090, 0.130],
                                tmaxs=[0.110, 0.150],
                                cluster_times=stats['w0_w15']['SII'])
plot_vertices(combination, max_vertices_dict['w0_w15'], 'PL',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['w0_w15']['PT'])
plot_vertices(combination, max_vertices_dict['w0_w15'], 'PI',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['w0_w15']['PI'])
plot_vertices(combination, max_vertices_dict['w0_w15'], 'TH',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['w0_w15']['TH'])
plot_vertices(combination, max_vertices_dict['w0_w15'], 'PL',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['w0_w15']['PL'])

#%% omission
combination = [
                dict(event='o0', first_event='o0', second_event='o15'),
                dict(event='o15', first_event='o0', second_event='o15'),
                ]

plot_vertices(combination, max_vertices_dict['o0_o15'], 'SI',
              ylim=(0, 48e-15), tmins=[0.030],
                                tmaxs=[0.070],
                                cluster_times=stats['o0_o15']['SI'])
plot_vertices(combination, max_vertices_dict['o0_o15'], 'CL6',
              ylim=(0, 48e-15), tmins=[0.100],
                                tmaxs=[0.120],
                                cluster_times=stats['o0_o15']['CL6'])
plot_vertices(combination, max_vertices_dict['o0_o15'], 'SII',
              ylim=(0, 48e-15), tmins=[0.090, 0.130],
                                tmaxs=[0.110, 0.150],
                                cluster_times=stats['o0_o15']['SII'])
plot_vertices(combination, max_vertices_dict['o0_o15'], 'PT',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['o0_o15']['PT'])
plot_vertices(combination, max_vertices_dict['o0_o15'], 'PI',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['o0_o15']['PI'])
plot_vertices(combination, max_vertices_dict['o0_o15'], 'TH',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['o0_o15']['TH'])
plot_vertices(combination, max_vertices_dict['o0_o15'], 'PL',
              ylim=(0, 65e-15), tmins=[0.030, 0.070, 0.110],
                                tmaxs=[0.060, 0.090, 0.130],
                                cluster_times=stats['o0_o15']['PL'])

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
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2'][0]['SI'], :])
    
    save_T1_plot_only(fig, join(save_path, 's1_SI_T1_' + weight_norm), time,
                      weight_norm, settings)
    
    #SII
    time = 0.123
    fig = stc.plot(src, initial_time=time, 
              clim=dict(kind='value', lims=(np.quantile(stc.data, 0.95),
                                  np.quantile(stc.data, 0.975),
                                  np.quantile(stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2'][0]['SII'], :])
    
    save_T1_plot_only(fig, join(save_path, 's1_SII_T1_' + weight_norm), time,
                      weight_norm, settings)
    
    
    ## save
    
    #Cerebellum 6 L
    time = 0.141
    fig = stc.plot(src, initial_time=time, 
              clim=dict(kind='value', lims=(np.quantile(stc.data, 0.95),
                                  np.quantile(stc.data, 0.975),
                                  np.quantile(stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2'][0]['CL6'], :])
    
    save_T1_plot_only(fig, join(save_path,
                                's1_Cerebellum_6_T1_' + weight_norm), time,
                       weight_norm, settings)
    
    # time = 0.124
    # fig = stc.plot(src, initial_time=time, 
    #           clim=dict(kind='value', lims=(np.quantile(stc.data, 0.99),
    #                               np.quantile(stc.data, 0.995),
    #                               np.quantile(stc.data, 0.999))),
    #           initial_pos=(-0.035, -0.051, -0.032))
    


#%%############################################################################
###############################################################################
### OSCILLATORY RESPONSES #####################################################
###############################################################################
###############################################################################

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
                   n_layers, excluded_subjects, fmin, fmax):
    for recording_index, recording in enumerate(recordings):
        subject = recording['subject']
        date = recording['date']
        if subject in bad_subjects or subject in excluded_subjects:
            continue
        print(subject)
        full_path = fname.source_hilbert_beamformer_simnibs_morph(
                subject=subject,
                date=date,
                fmin=fmin, fmax=fmax,
                tmin=-0.750, tmax=0.750,
                event=combination['contrast'][0]['event'],
                first_event=combination['contrast'][0]['first_event'],
                second_event=combination['contrast'][0]['second_event'],
                reg=0.00, weight_norm='unit-noise-gain-invariant',
                n_layers=n_layers)
        full_path2 = fname.source_hilbert_beamformer_simnibs_morph(
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


data = dict()
data['s1_s2'] = dict()
data['w0_w15'] = dict()
data['o0_o15'] = dict()

fs = [
      # (4, 7),
      # (8, 12),
       (14, 30)
      ]

comparisons = (
                  's1_s2',
                  'w0_w15',
                  'o0_o15'
              )

for f in fs:
    
    f_text = str(f[0]) + '_' + str(f[1]) + '_Hz'
    for comparison_index, comparison in enumerate(comparisons):
        print(f_text + ': ' +  comparison)
        data[comparison][f_text] = \
            load_data_stat(combinations[comparison_index],
                           recordings, bad_subjects, 3,
                           subjects_with_no_BEM_simnibs, f[0], f[1])
   


#%% PLOT VERTICES OSCILLATORY

plt.close('all')

def plot_vertices_oscillatory(combinations, vertex_dict, roi_name, ylim,
                      fmin, fmax, cluster_times=None, xlim=None,
                      save=True,
                      subject='fsaverage',
                      date='20210825_000000'):
    f_text = str(fmin) + '_' + str(fmax) + '_Hz'

    fig = plt.figure()
    fig.set_size_inches(8, 6)
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['font.weight'] = 'bold'
    legends = list()

    save_path = fname.subject_figure_path(subject=subject, date=date)
    
    these_data = list()
    
    for combination_index, combination in enumerate(combinations):
        ga_path = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject=subject,
        date=date,
        fmin=fmin, fmax=fmax, tmin=-0.750, tmax=0.750, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm='unit-noise-gain-invariant',
                    n_layers=3)
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
 


        # stc.crop(tmin=-0.050, tmax=0.400) ## cropping
        if (combination_index == 0 and combination['event'] == 's1') or \
           (combination_index == 0 and combination['event'] == 'w0') or \
           (combination_index == 0 and combination['event'] == 'o0'):
               pass
            # time_peaks, peaks = find_peaks(stc, tmins, tmaxs, vertex_index)
        # stc._data = np.abs(stc.data) ## taking abs
        if type(vertex_dict) is int:
            stc_index = np.where(stc.vertices[0]  == vertex_dict)[0][0]
        elif type(vertex_dict) is list:
            stc_indices = list()
            n_vertices = len(vertex_dict)
            
            for i in range(n_vertices):
                # print(vertex_dict[i])
                # print(stc.vertices[0])

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
        
    # plt.ylim(ylim[0], ylim[1])
    plt.legend(legends)
    # plt.vlines(time_peaks, ylim[0], peaks, linestyles='dashed',
    #            color='k')
    # set_texts(time_peaks, peaks)
    plt.xlabel('Time (ms)')
    plt.ylabel('Source strength, unit-noise-gain (T)')
    if ylim is None:
        ylim = fig.get_axes()[0].get_ylim()
    plt.vlines(0, ylim[0], ylim[1], linestyles='dashed',
               color='k', linewidth=3)
    roi_name = roi_rewriting(roi_name)
    plt.title(roi_name)
    if cluster_times is not None:
        if len(cluster_times) > 0:
            # print(cluster_times)
            for cluster in cluster_times:
                # for cluster_time_index, cluster_time in enumerate(cluster):
                # print(cluster_time)
                cluster_index_begin = find_nearest(stc.times, cluster[0])
                cluster_index_end   = find_nearest(stc.times, cluster[-1])
                # print((cluster_index_begin, cluster_index_end))
                # print(these_data[0][cluster_index_begin:(cluster_index_end+1)])
                # print(np.unique(cluster))
                plt.fill_between(np.unique(cluster) * 1e3,
                      these_data[0][cluster_index_begin:(cluster_index_end+1)],
                      these_data[1][cluster_index_begin:(cluster_index_end+1)],
                        color=colours[0], alpha=0.5)
    ## stats
   
    if xlim is not None:
        plt.xlim(xlim)
    plt.show()
    filename = 'oscillatory_stc_' + combination['event'] + '_' +  \
        f_text + '_' + roi_name.replace(' ', '_')
    if save:
        fig.savefig(join(save_path, filename), dpi=300)


    
#%% FIND LABEL VERTICES

def find_label_vertices(stc, label_dict, stc_full_path, src):
    vertices = src[0]['vertno']
    nifti_full_path = stc_full_path[:-2] + 'nii'
    # stc.save_as_volume(nifti_full_path, src, overwrite=True)
    img = image.load_img(nifti_full_path)
    data = np.asanyarray(img.dataobj)
    label_indices = dict()
    src_indices = dict()

    for roi, DICT in label_dict.items():
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

label_dict = dict(
                CL6=dict(label='Cerebelum_6_L', atlas='AAL',
                          restrict_time_index=None),
                CL1=dict(label='Cerebelum_Crus1_L', atlas='AAL',
                          restrict_time_index=None),
                SI= dict(label='Postcentral_L', atlas='AAL',
                          restrict_time_index=None),
                SII=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=None, harvard_side='L',
                          boolean_index=10),
                SII_R=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=None, harvard_side='R',
                          boolean_index=10),
                PT=dict(label='Putamen_L', atlas='AAL',
                          restrict_time_index=None),
                PI=dict(label='Parietal_Inf_L', atlas='AAL',
                          restrict_time_index=None),
                TH=dict(label='Thalamus_L', atlas='AAL',
                          restrict_time_index=None),
                TH_R=dict(label='Thalamus_R', atlas='AAL',
                          restrict_time_index=None),
                PL=dict(label='Pallidum_L', atlas='AAL',
                            restrict_time_index=None),
                SMA=dict(label='Supp_Motor_Area_L', atlas='AAL',
                          restrict_time_index=None),
                MI=dict(label='Precentral_L', atlas='AAL',
                          restrict_time_index=None),
                # INS=dict(label='Insual_L', atlas='AAL',
                          # restrict_time_index=None)
                )

stc_full_path = \
        fname.source_hilbert_beamformer_grand_average_simnibs(
            subject='fsaverage', date='20210825_000000', fmin=14, fmax=30,
            tmin=-0.750, tmax=0.750, reg=0.00,
            event='o0', first_event='o0', second_event='o15',
            weight_norm='unit-noise-gain-invariant', n_layers=3)
stc = mne.read_source_estimate(stc_full_path)
src = mne.read_source_spaces(fname.anatomy_volumetric_source_space(
                            subject='fsaverage', spacing=7.5))

stc_indices, src_indices = find_label_vertices(stc, label_dict, stc_full_path,
                                               src)

#%%# how to calculate spatial adjacency 

def reduce_adjacency(src, n_times, vertices):

    from scipy.sparse import coo_matrix
    print(n_times)

    adj = mne.spatio_temporal_src_adjacency(src, 1) ## downsize this to label...
    adj_arr = adj.toarray()
    adj_arr = adj_arr[vertices, :] ## what goes in the second slice...
    n_vertices = adj_arr.shape[0]
    
    adj_reduced = np.zeros((n_vertices, n_vertices))
    
    for vertex_index in range(n_vertices):
        adj_reduced[vertex_index, :] = adj_arr[vertex_index,
                                               vertices]
        
    if n_times > 1:
        final_adj = np.zeros(shape=(n_times * n_vertices,
                                    n_times * n_vertices))
        for time_index in range(n_times):
            start_index = time_index * n_vertices
            end_index = (time_index + 1) * n_vertices
            final_adj[start_index:end_index, ## copy matrix
                      start_index:end_index] = adj_reduced
            
            ## add side lines # don't get them completely, but maybe time dep?
            start_index = time_index * n_vertices
            middle_index = (time_index + 1) * n_vertices
            end_index = (time_index + 2) * n_vertices
         
            if time_index < (n_times-1):
                ## first line
                final_adj[start_index:middle_index,
                          middle_index:end_index] = np.identity(n_vertices)
                ## second line
                final_adj[middle_index:end_index,
                          start_index:middle_index] = np.identity(n_vertices)
    else:
        final_adj = adj_reduced
        
    adj_coo = coo_matrix(final_adj)
    
    return adj_coo
    
#%% try cluster


def get_clusters_label(data, tmin, tmax, stc, vertex_indices, alpha, roi, src,
                       tail=0):
    tmin_index = find_nearest(stc.times, tmin)
    tmax_index = find_nearest(stc.times, tmax)
    
    print((tmin_index, tmax_index))
    
    adj = reduce_adjacency(src, tmax_index - tmin_index, 
                           vertex_indices[roi])
    
    # plt.figure()
    # plt.spy(adj)
    # plt.show()


    test_data = data[:, vertex_indices[roi], tmin_index:tmax_index]
    # print(test_data.shape)
    test_data = np.swapaxes(test_data, 1, 2)
    # print(test_data.shape)

        
    test_times = stc.times[tmin_index:tmax_index]
    t_obs, clusters, cluster_pv, H0 = \
        mne.stats.spatio_temporal_cluster_1samp_test(test_data,
                                                      n_permutations=1e3,
                                                 seed=7, adjacency=adj,
                                                 tail=tail, n_jobs=4)
    
    cluster_indices = np.where(cluster_pv < alpha)[0]
    sig_cluster_pv = cluster_pv[cluster_indices]
    print(np.sort(cluster_pv))
    cluster_times = list()
    for cluster_index in cluster_indices:
        cluster_time = test_times[clusters[cluster_index][0]]
        cluster_times.append(cluster_time)
        
    return sig_cluster_pv, cluster_times
              



#%% FULL PLOT

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 8
mpl.rcParams['font.weight'] = 'bold'
plt.close('all')

def get_cluster_times_label(stats, stc_indices, roi, alpha=0.05):
    clusters = stats['clusters']
    cluster_pv = stats['cluster_p_values']
    
    
    cluster_times = list()
    for cluster_index, cluster in enumerate(clusters):
        if cluster_pv[cluster_index] >= alpha:
            continue
        # print(cluster_index)
        cluster_time_indices = cluster[0]
        cluster_stc_indices  = cluster[1]
        check_counter = 0
        for counter, cluster_stc_index in enumerate(cluster_stc_indices):
            if cluster_stc_index in stc_indices[roi]:
                # print(check_counter)
                check_counter += 1
                cluster_times.append(cluster_time_indices[counter])
                
    cluster_times = np.array(cluster_times)
    # print(len(cluster_times))
    cluster_times = np.unique(cluster_times)
    cluster_times = cluster_times / 1e3
    # print(cluster_times)
    
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

            these_clusters.append(cluster_times[this_slice])#,
                                        # cluster_times[this_cluster_end_index],
                                        # 0.001))  
            if (time_index + 2) <= len(cluster_times):
                next_cluster_begin_index = time_index + 1
  
  
            
    return these_clusters

        

#%%

def save_T1_plot_only_oscillatory(fig, filename, time):
    mpl.rcParams['font.size'] = 8
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    axis_0 = fig.axes[0]
    extent = \
        axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if time is not None:
        cbar_text = axis_0.get_children()[-2]
        cbar_text.set_text('Power change: (x$_1$ - x$_2$) / (x$_1$ + x$_2$) ' + \
                           str(round(time * 1e3, 1)) + ' ms')
        cbar_text.set_fontsize('small')
    fig.savefig(filename, dpi=300, bbox_inches=extent.expanded(1.05, 1.17))

def plot_full_cluster(comparison, fmin, fmax, stat_tmin, stat_tmax,
                      roi=None, stc_indices=stc_indices,
                      src_indices=None, src=None,
                      pos=None, time=None, mode='stat_map',
                      lims=None):
    mpl.rcParams['font.size'] = 8

    events = comparison.split('_')

    stc_full_path_1 = \
            fname.source_hilbert_beamformer_grand_average_simnibs(
                subject='fsaverage', date='20210825_000000', fmin=fmin,
                fmax=fmax,
                tmin=-0.750, tmax=0.750, reg=0.00,
                event=events[0], first_event=events[0], second_event=events[1],
                weight_norm='unit-noise-gain-invariant', n_layers=3)
            
    stc_1 = mne.read_source_estimate(stc_full_path_1)
    
    stc_full_path_2 = \
            fname.source_hilbert_beamformer_grand_average_simnibs(
                subject='fsaverage', date='20210825_000000', fmin=fmin,
                fmax=fmax,
                tmin=-0.750, tmax=0.750, reg=0.00,
                event=events[1], first_event=events[0], second_event=events[1],
                weight_norm='unit-noise-gain-invariant', n_layers=3)
            
    stc_2 = mne.read_source_estimate(stc_full_path_2)

    ratio = stc_1.copy()
    ratio._data = (stc_1.data - stc_2.data) / (stc_1.data + stc_2.data)
    ratio.crop(stat_tmin, stat_tmax)
    
    stats_filename = fname.source_hilbert_beamformer_statistics(
        subject='fsaverage', date='20210825_000000', fmin=fmin, fmax=fmax,
        tmin=-0.750, tmax=0.750, reg=0.00, first_event=events[0],
        second_event=events[1], stat_tmin=stat_tmin, stat_tmax=stat_tmax,
        nperm=1024, seed=7, condist=None, pval=0.05)

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


#%% PLOT IT ALL

def plot_it_all(x1, x2, fmin, fmax, stat_tmin, stat_tmax, roi=None,
                stc_indices=None, src_indices=None, src=None,
                xlim=None, time=None, mode='stat_map',
                lims=None):
    comparison = x1 + '_' + x2
    f_text = str(fmin) + '_'  + str(fmax) + '_Hz'
    fig, stats, time = plot_full_cluster(comparison, fmin, fmax, stat_tmin,
                                   stat_tmax,
                                   roi=roi, stc_indices=stc_indices,
                                   src_indices=src_indices, src=src,
                                   mode=mode, lims=lims, time=time)
    T1_filename = comparison + '_' + f_text + '_' + roi
    if mode =='glass_brain':
        T1_filename = T1_filename + '_' + mode
    print('Filename: ' + T1_filename)
    figure_path = fname.subject_figure_path(subject='fsaverage',
                                            date='20210825_000000')

    if T1_filename is not None:
        save_T1_plot_only_oscillatory(fig, join(figure_path, T1_filename),
                                      time=time)
    cluster_times = get_cluster_times_label(stats, stc_indices, roi)
    plot_vertices_oscillatory(get_combination(x1, x2),
                              src_indices[roi], roi, None,
                              fmin, fmax, cluster_times=cluster_times,
                              xlim=xlim)
#%% Call it - big loop

plt.close('all')

freqs = [
    (4, 7),
    (14, 30)
    ]
comparisons = [
    ('s1', 's2'),
    ('w0', 'w15'),
    ('o0', 'o15')
    ]
rois = [
        'SI', 'SII', 'CL6', 'PT', 'TH',
        'PI', 'PL'
        ]

for freq in freqs:
    for comparison in comparisons:
        for roi in rois:
            plot_it_all(comparison[0], comparison[1], freq[0], freq[1],
                        0.000, 0.200, roi=roi,
                        stc_indices=stc_indices, src_indices=src_indices,
                        src=src)            


#%% single calls
plot_it_all('s1', 's2', 4, 7, 0.000, 0.200, roi='SI',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.020, 0.025, 0.030))
plot_it_all('s1', 's2', 4, 7, 0.000, 0.200, roi='CL6',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.020, 0.025, 0.030))   
plot_it_all('s1', 's2', 4, 7, 0.000, 0.200, roi='SII',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.020, 0.025, 0.030))
plot_it_all('s1', 's2', 4, 7, 0.000, 0.200, roi='TH',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.020, 0.025, 0.030)) 

plot_it_all('s1', 's2', 14, 30, 0.000, 0.200, roi='SI',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.030, 0.035, 0.040))
plot_it_all('s1', 's2', 14, 30, 0.000, 0.200, roi='SII',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.019, 0.021, 0.023))

plot_it_all('s1', 's2', 14, 30, 0.000, 0.200, roi='SII_R',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.019, 0.021, 0.023)) 
plot_it_all('s1', 's2', 14, 30, 0.000, 0.200, roi='TH',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.019, 0.020, 0.023))


plot_it_all('s1', 's2', 14, 30, 0.000, 0.200, roi='PT',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.019, 0.020, 0.021))
plot_it_all('s1', 's2', 14, 30, 0.000, 0.200, roi='TH_R',
            stc_indices=stc_indices, src_indices=src_indices, src=src,
            lims=(0.0185, 0.020, 0.022))

# plot_it_all('s1', 's2', 14, 30, 0.000, 0.200, roi='CL6',
#             stc_indices=stc_indices, src_indices=src_indices, src=src,
#             lims=(0.010, 0.012, 0.014))





plot_it_all('o0', 'o15', 14, 30, 0.000, 0.200, roi='SI',
            stc_indices=stc_indices, src_indices=src_indices, src=src)
plot_it_all('o0', 'o15', 14, 30, 0.000, 0.200, roi='SII',
            stc_indices=stc_indices, src_indices=src_indices, src=src) 


plot_it_all('o0', 'o15', 14, 30, 0.000, 0.200, roi='TH',
            stc_indices=stc_indices, src_indices=src_indices, src=src) 

plot_it_all('o0', 'o15', 14, 30, 0.000, 0.200, roi='CL6',
            stc_indices=stc_indices, src_indices=src_indices, src=src)    


#%% find label based on vertex

# atlas = 'AAL'
# # def find_label_for_vertex(stc, vertex, stc_full_path, src, atlas):
# vertices = src[0]['vertno']
# nifti_full_path = stc_full_path[:-2] + 'nii'
# if not exists(nifti_full_path):
#     stc.save_as_volume(nifti_full_path, src, overwrite=True)
# img = image.load_img(nifti_full_path)
# # data = np.asanyarray(img.dataobj)
# if atlas == 'AAL':
#     atlas = datasets.fetch_atlas_aal()
# atlas_img = image.load_img(atlas['maps'])
# atlas_interpolated = image.resample_to_img(atlas_img, img, 'nearest')
# atlas_interpolated_data = np.asanyarray(atlas_interpolated.dataobj)  
# all_labels = atlas['labels']
# all_indices = atlas['indices']


# rois = list()
# for stc_index in np.unique(c_stc_indices):
#     src_index = vertices[stc_index]
#     label_index = atlas_interpolated_data.flatten()[src_index]
#     if label_index > 0:
#         for index in all_indices:
#             if label_index == int(index):
#                 this_index = all_indices.index(str(label_index))
#                 roi = all_labels[this_index]
#                 if roi not in rois:
#                     rois.append(roi)
                    
# rois.sort()
             

#%% FULL BRAIN PLOT (THETA)

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
    unit_noise_gain_invariant='Power change (%) (x1 - x2) / (x1 + x2) ')

fmin = 4
fmax = 7
f_text = str(fmin) + '_' + str(fmax) + '_Hz'

for weight_norm in weight_norms:

    ga_path_1 = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject=subject,
        date=date,
        fmin=4, fmax=7, tmin=-0.750, tmax=0.750, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm=weight_norm,
                    n_layers=3)
    ga_path_2 = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject=subject,
        date=date,
        fmin=4, fmax=7, tmin=-0.750, tmax=0.750, reg=0.00,
        event=combination['second_event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm=weight_norm,
                    n_layers=3)
    
    
    stc_1 = mne.read_source_estimate(ga_path_1)
    stc_2 = mne.read_source_estimate(ga_path_2)
    
    stc = stc_1.copy()
    stc._data = (stc_1.data - stc_2.data) / (stc_1.data + stc_2.data)
    stc._data *= 1e2 # change into percent
    
    
    ## SI
    time = 0.150
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', pos_lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2'][f_text]['SI'], :])
    
    save_T1_plot_only(fig, join(save_path, 's1_s2_hilbert_SI_T1_' + weight_norm), time,
                      weight_norm, settings)   

    ## SII
    time = 0.175
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', pos_lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2'][f_text]['SII'], :])
    
    save_T1_plot_only(fig, join(save_path, 's1_s2_hilbert_SII_T1_' + weight_norm), time,
                      weight_norm, settings)

    ## Inferior Parietal
    time = 0.142
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', pos_lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2'][f_text]['PI'], :])
    
    save_T1_plot_only(fig, join(save_path, 's1_s2_hilbert_PI_T1_' + weight_norm), time,
                      weight_norm, settings)
    
    ## Cerebellum 6
    time = 0.100
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', pos_lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['s1_s2'][f_text]['CL6'], :])
    
    save_T1_plot_only(fig, join(save_path, 's1_s2_hilbert_CL6_T1_' + weight_norm), time,
                      weight_norm, settings)

#%% OMISSION BETA       

plt.close('all')

combination = dict(event='o0', first_event='o0', second_event='o15')
weight_norms = [
    # 'unit-gain',
    'unit-noise-gain-invariant'
    ]
settings = dict(
    unit_gain='First Stimulation: Source strength, unit gain (Am) at: ',
    unit_noise_gain_invariant='Power change (%) (x1 - x2) / (x1 + x2) ')

fmin = 14
fmax = 30
f_text = str(fmin) + '_' + str(fmax) + '_Hz'

for weight_norm in weight_norms:

    ga_path_1 = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject=subject,
        date=date,
        fmin=fmin, fmax=fmax, tmin=-0.750, tmax=0.750, reg=0.00,
        event=combination['event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm=weight_norm,
                    n_layers=3)
    ga_path_2 = fname.source_hilbert_beamformer_grand_average_simnibs(
        subject=subject,
        date=date,
        fmin=fmin, fmax=fmax, tmin=-0.750, tmax=0.750, reg=0.00,
        event=combination['second_event'], first_event=combination['first_event'],
                    second_event=combination['second_event'],
                    weight_norm=weight_norm,
                    n_layers=3)
    
    
    stc_1 = mne.read_source_estimate(ga_path_1)
    stc_2 = mne.read_source_estimate(ga_path_2)
    
    stc = stc_1.copy()
    stc._data = (stc_1.data - stc_2.data) / (stc_1.data + stc_2.data)
    stc._data *= 1e2 # change into percent
    
    
    ## CL1
    time = 0.037
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', pos_lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['o0_o15'][f_text]['CL1'], :])
    
    save_T1_plot_only(fig, join(save_path, 'o0_o15_hilbert_CL1_T1_' + weight_norm), time,
                      weight_norm, settings) 
    
    ## CL6
    time = 0.131
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', pos_lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['o0_o15'][f_text]['CL6'], :])
    
    save_T1_plot_only(fig, join(save_path, 'o0_o15_hilbert_CL6_T1_' + weight_norm), time,
                      weight_norm, settings)  
    
    
    ## SI
    time = 0.150
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', pos_lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['o0_o15'][f_text]['SI'], :])
    
    save_T1_plot_only(fig, join(save_path, 'o0_o15_hilbert_SI_T1_' + weight_norm), time,
                      weight_norm, settings) 
    
    ## SII
    time = 0.083
    this_stc = stc.copy()
    # this_stc.crop(tmax=0.060)
    fig = this_stc.plot(src, initial_time=time,
              clim=dict(kind='value', pos_lims=(np.quantile(this_stc.data, 0.95),
                                            np.quantile(this_stc.data, 0.975),
                                            np.quantile(this_stc.data, 0.99))),
              initial_pos=src[0]['rr'][max_vertices_dict['o0_o15'][f_text]['SII'], :])
    
    save_T1_plot_only(fig, join(save_path, 'o0_o15_hilbert_SII_T1_' + weight_norm), time,
                      weight_norm, settings) 
    
    #%% BIG LOOP

    mr_path = fname.subject_bem_path(subject='fsaverage')

    max_vertices_dict = dict(
        s1_s2=dict(),
        w0_w15=dict(),
        o0_o15=dict()
        )
    stats = dict(
                    s1_s2=dict(),
                    w0_w15=dict(),
                    o0_o15=dict()
                  )

    fs = (
          (4, 7),
          # (8, 12),
          (14,30)
          )
    for f in fs:
        
        f_text = str(f[0]) + '_' + str(f[1]) + '_Hz'
        for comparison in stats.keys():
            stats[comparison][f_text] = dict()
            max_vertices_dict[comparison][f_text] = dict()

    src = mne.read_source_spaces(join(mr_path, 'volume-7.5_mm-src.fif'))

    for comparison_index, comparison in enumerate(stats):
        event = combinations[comparison_index]['contrast'][0]['event']
        first_event = combinations[comparison_index]['contrast'][0]['first_event']
        second_event = combinations[comparison_index]['contrast'][0]['second_event']
        
        fs = (
            (4, 7),
            # (8, 12),
            (14,30)
            )

        for f in fs:
            
            f_text = str(f[0]) + '_' + str(f[1]) + '_Hz'

            ## find mean - don't bias
            for label in label_dict: 
         
                print('\n' + comparison + ' ' + label +  '\n' + f_text + '\n')
                ## FIXME: think about how to incorporate more than one time range
                stats[comparison][f_text][label] = \
                    get_clusters_label(data[comparison][f_text], 0.000, 0.200,
                                       stc,
                                  stc_indices, roi=label,
                                  alpha=0.05, src=src)

#%% clust and plot 
    


def clust_and_plot(data, comparison, fmin, fmax, tmin, tmax, roi, stc,
                   stc_indices, src_indices, src, alpha):
    
    events = comparison.split('_')
    
    f_text = str(fmin) + '_'  + str(fmax) + '_Hz'
    sig_cluster_pv, cluster_times = \
        get_clusters_label(data[comparison][f_text], tmin, tmax,
                            stc, stc_indices, alpha=alpha, roi=roi, src=src)
    combination = [
        dict(event=events[0], first_event=events[0], second_event=events[1]),
        dict(event=events[1], first_event=events[0], second_event=events[1]),
                   ]
    plot_vertices_oscillatory(combination,
                              src_indices[roi], roi,
                              None, fmin, fmax, cluster_times)
    
    
#%% CL6 omission theta **

clust_and_plot(data, 'o0_o15', 4, 7, 0.000, 0.200, 'CL6',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% CL6 omission beta **
clust_and_plot(data, 'o0_o15', 14, 30, 0.020, 0.070, 'CL6',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% SI omission beta 
clust_and_plot(data, 'o0_o15', 14, 30, -0.400, -0.300, 'SI',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% MI omission theta **
clust_and_plot(data, 'o0_o15', 4, 7, 0.200, 0.400, 'MI',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% SMA weak beta 
clust_and_plot(data, 'w0_w15', 14, 30, 0.300, 0.400, 'SMA',
                stc, stc_indices, src_indices, src, alpha=0.05)

#%% SMA weak beta 
clust_and_plot(data, 'o0_o15', 14, 30, 0.300, 0.400, 'SMA',
                stc, stc_indices, src_indices, src, alpha=0.05)

#%% WEAK SII theta **
        
clust_and_plot(data, 'w0_w15', 14, 30, 0.000, 0.150, 'SII',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% WEAK PL theta **
        
clust_and_plot(data, 'w0_w15', 4, 7, 0.000, 0.100, 'PL',
               stc, stc_indices, src_indices, src, alpha=0.05)


#%% WEAK CL theta **
        
clust_and_plot(data, 'w0_w15', 4, 7, 0.180, 0.340, 'CL6',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% TH omission beta 
clust_and_plot(data, 'o0_o15', 14, 30,-0.020, 0.020, 'TH',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% TH weak beta 
clust_and_plot(data, 'w0_w15', 14, 30,-0.020, 0.020, 'TH',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% SII omission beta 
clust_and_plot(data, 'o0_o15', 4, 7, -0.050, 0.150, 'SII',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% SI omission theta 
clust_and_plot(data, 'o0_o15', 4, 7, 0.000, 0.150, 'SI',
               stc, stc_indices, src_indices, src, alpha=0.05)


#%% PI omission theta **
clust_and_plot(data, 'o0_o15', 4, 7, -0.400, -0.200, 'PI',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% PI omission theta 
clust_and_plot(data, 'w0_w15', 4, 7, -0.000, 0.200, 'PI',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% SI stim beta
clust_and_plot(data, 's1_s2', 14, 30, -0.000, 0.200, 'SI',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% SII stim beta
clust_and_plot(data, 's1_s2', 14, 30, -0.000, 0.200, 'SII',
               stc, stc_indices, src_indices, src, alpha=0.05)

#%% CL6 stim beta
clust_and_plot(data, 's1_s2', 14, 30, 0.550, 0.750, 'CL6',
               stc, stc_indices, src_indices, src, alpha=0.05)  