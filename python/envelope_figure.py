#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:08:19 2022

@author: lau
"""

#%% IMPORTS

from os import chdir
script_path = '/home/lau/projects/functional_cerebellum/scripts/python/'
chdir(script_path)

from config import (fname, recordings, bad_subjects, 
                    subjects_with_no_BEM_simnibs)

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from os.path import join
from scipy.stats import t
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

src = mne.read_source_spaces(fname.anatomy_volumetric_source_space(
                                subject='fsaverage', spacing=7.5))


fig_path = fname.subject_figure_path(subject='fsaverage',
                                      date='20210825_000000')
n_sources = len(src[0]['vertno'])
n_subjects = len(recordings)

data = np.zeros(shape=(n_subjects, n_sources, n_sources))


#%% FUNCTIONS

def get_data_array(fmin, fmax, tmin, tmax, event, data):
    
    print('Loading for range: ' + str(fmin) + '-' + str(fmax) + \
          ' Hz for event: ' + event)
    indices = list()
    for recording_index, recording in enumerate(recordings[:]):
        subject = recording['subject']
        # if subject == '0005' or subject == '0008' or subject == '0015' or \
        #     subject == '0016' or subject == '0017' or subject == '0018':
        #     continue
        if subject in bad_subjects:
            continue
        if subject in subjects_with_no_BEM_simnibs:
            continue
        date    = recording['date']
        first_event = event[0] + '0'
        second_event = event[0] + '15'
        
        filename = fname.envelope_correlation_morph_data(
                   subject=subject, date=date, fmin=fmin, fmax=fmax,
                   tmin=tmin, tmax=tmax, reg=0.00,
                   weight_norm='unit-noise-gain-invariant',
                   event=event, n_layers=3,
                   first_event=first_event, second_event=second_event)
        
        print('Loading subject: ' + subject)
        this_data = np.load(filename)
        data[recording_index, :, :] = this_data
        indices.append(recording_index)

        
    return data, indices


from nilearn import image, datasets

def find_label_vertices(stc, label_dict, stc_full_path, src):
    vertices = src[0]['vertno']
    nifti_full_path = stc_full_path[:-2] + 'nii'
    img = image.load_img(nifti_full_path)
    data = np.asanyarray(img.dataobj)
    label_indices = dict()

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
            x, y, z, ts = \
                np.where(
            np.max(label_data[:, :, :, :DICT['restrict_time_index']]) == \
                label_data) 
        else:
            x, y, z = np.where(label_data[:, :, :, 0] > 0)
        
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
    return label_indices


#%% GET DATA
        
# theta_w0, indices = get_data_array(4, 7, -0.100, 0.100, 'w0', data)
# theta_w0 = theta_w0[indices, :, :]
# theta_w15, indices = get_data_array(4, 7, -0.100, 0.100, 'w15', data)
# theta_w15 = theta_w15[indices, :, :]

# theta_o0, indices = get_data_array(4, 7, -0.100, 0.100, 'o0', data)
# theta_o0 = theta_o0[indices, :, :]
# theta_o15, indices = get_data_array(4, 7, -0.100, 0.100, 'o15', data)
# theta_o15 = theta_o15[indices, :, :]

beta_w0, indices = get_data_array(14, 30, -0.100, 0.100, 'w0', data)
beta_w0 = beta_w0[indices, :, :]
beta_w15, indices = get_data_array(14, 30, -0.100, 0.100, 'w15', data)
beta_w15 = beta_w15[indices, :, :]

beta_o0, indices = get_data_array(14, 30, -0.100, 0.100, 'o0', data)
beta_o0 = beta_o0[indices, :, :]
beta_o15, indices = get_data_array(14, 30, -0.100, 0.100, 'o15', data)
beta_o15 = beta_o15[indices, :, :]

#%% FULL LABEL DICT
stc_full_path = \
        fname.source_hilbert_beamformer_grand_average_simnibs(
            subject='fsaverage', date='20210825_000000', fmin=14, fmax=30,
            tmin=-0.750, tmax=0.750, reg=0.00,
            event='o0', first_event='o0', second_event='o15',
            weight_norm='unit-noise-gain-invariant', n_layers=3)
stc = mne.read_source_estimate(stc_full_path)
src = mne.read_source_spaces(fname.anatomy_volumetric_source_space(
                            subject='fsaverage', spacing=7.5))
atlas = datasets.fetch_atlas_aal()
labels = atlas['labels']

label_dict = dict()
for label in labels:
    label_dict[label] = dict(label=label, atlas='AAL', 
                             restrict_time_index=None)

all_vertex_indices = find_label_vertices(stc, label_dict, stc_full_path, src)

#%% FIND SPECIFIC LABEL VERTICES

label_dict = dict(
                Postcentral_L=  dict(label='Postcentral_L', atlas='AAL',
                        restrict_time_index=None),
                Postcentral_R=  dict(label='Postcentral_R', atlas='AAL',
                        restrict_time_index=None),
                # Precentral_L=  dict(label='Precentral_L', atlas='AAL',
                #         restrict_time_index=None),
                # Precentral_R=  dict(label='Precentral_R', atlas='AAL',
                #         restrict_time_index=None),
                # Parietal_Operculum_Cortex= \
                #     dict(label='Parietal Operculum Cortex', atlas='harvard',
                #           restrict_time_index=None),
                Cerebelum_6_L= dict(label='Cerebelum_6_L', atlas='AAL',
                         restrict_time_index=None),
                Cerebelum_6_R= dict(label='Cerebelum_6_R', atlas='AAL',
                         restrict_time_index=None),
                # Pallidum_L= dict(label='Pallidum_L', atlas='AAL',
                #           restrict_time_index=None),
                # Pallidum_R= dict(label='Pallidum_R', atlas='AAL',
                #           restrict_time_index=None),
                Thalamus_L= dict(label='Thalamus_L', atlas='AAL',
                          restrict_time_index=None),
                Thalamus_R= dict(label='Thalamus_R', atlas='AAL',
                          restrict_time_index=None),
                )



vertex_indices = find_label_vertices(stc, label_dict, stc_full_path, src)

#%% ANOTHER T TRY

def get_sig (a1, a2, seed, atlas, vertex_indices, n=26):
    
    ts = list()
    values = dict()
    # cutoff = abs(t.ppf(0.025, n-1))
    # for label in atlas['labels']:
    for label in vertex_indices:
        this_diff = \
        a1[:, vertex_indices[seed], :][:, :, vertex_indices[label]] - \
        a2[:, vertex_indices[seed], :][:, :, vertex_indices[label]]
        
        mu = np.nanmean(np.mean(np.median(this_diff, axis=2), axis=1))
        sem = np.nanstd(np.mean(np.median(this_diff, axis=2), axis=1)) / \
            np.sqrt(n)
        
        # print(mu / sem)
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
    # for label_index, label in enumerate(atlas['labels']):
    for label_index, label in enumerate(vertex_indices):

        ts_dict[label] = ts[label_index]
    
    # print(np.array(atlas['labels'])[ts < -cutoff])
    # print(ts[ts < -cutoff])
    # print(np.array(atlas['labels'])[ts > cutoff])
    # print(ts[ts > cutoff])
    
    return ts_dict, values

def get_ps(ts_dict, alpha, n, fdr_correct=False):
    ps = dict()

    for roi in ts_dict:
        tval = ts_dict[roi]
        ps[roi] = t.sf(np.abs(tval), n-1) * 2
        
    return ps
        
def print_ps(ps):
    for roi in ps:
        if ps[roi] < 0.05:
            print(roi)
            print(ps[roi])
            

        
# ts = print_sig(theta_w0, theta_w15, 'Cerebelum_6_L', atlas, all_vertex_indices)
# ps = get_ps(ts, 0.05, 24)
# ts = print_sig(theta_o0, theta_o15, 'Cerebelum_6_L', atlas)

# ts_sub_L = print_sig(theta_w0, theta_w15, 'Cerebelum_6_L', atlas, vertex_indices)


ts_w, values_w = get_sig(beta_w0, beta_w15, 'Cerebelum_6_L', atlas, all_vertex_indices)
# ts_sub_w, diff = get_sig(beta_w0, beta_w15, 'Cerebelum_6_L', atlas, vertex_indices)
# ts_sub_R = print_sig(beta_w0, beta_w15, 'Cerebelum_6_R', atlas, all_vertex_indices)
# lala = print_sig(beta_w0, beta_w15, 'Thalamus_R', atlas, vertex_indices)

ts_o, values_o = get_sig(beta_o0, beta_o15, 'Cerebelum_6_L', atlas, all_vertex_indices)

print('weak beta')
print_ps(get_ps(ts_w, 0.05, 26))
print('\nomission beta')
print_ps(get_ps(ts_o, 0.05, 26))
    
    
#%% ANOVA

def do_anova(values_w, values_o, roi, plot=True, handle_nans=False):
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
        mpl.rcParams.update(mpl.rcParamsDefault)
        mpl.rcParams['font.size'] = 14
        mpl.rcParams['font.weight'] = 'bold'
        mpl.rcParams['lines.linewidth'] = 3
        means = np.array([np.mean(this_w_no_jitter), np.mean(this_w_jitter),
                 np.mean(this_o_no_jitter), np.mean(this_o_jitter)])
        
        sems = np.array(
            [np.std(this_w_no_jitter)  / np.sqrt(len(this_w_no_jitter)),
                np.std(this_w_jitter)  / np.sqrt(len(this_w_jitter)),
                np.std(this_o_no_jitter)  / np.sqrt(len(this_o_no_jitter)),
                np.std(this_o_jitter)  / np.sqrt(len(this_o_jitter))]
            )
        
        if handle_nans:
            mpl.rcParams.update(mpl.rcParamsDefault)
            mpl.rcParams['font.size'] = 14
            mpl.rcParams['font.weight'] = 'bold'
            mpl.rcParams['lines.linewidth'] = 3
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
            plt.ylim(0.2875, 0.3025)
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
        fig.savefig(join(fig_path,
                         'envelope_ANOVA_' + roi + '.png'), dpi=300)
    

do_anova(values_w, values_o, 'Thalamus_R')
do_anova(values_w, values_o, 'Thalamus_L')
# do_anova(values_w, values_o, 'Precentral_L', handle_nans=True)
# do_anova(values_w, values_o, 'Postcentral_R', handle_nans=True)
# do_anova(values_w, values_o, 'Insula_R', handle_nans=False)


#%% all plots
plt.close('all')
for label in atlas['labels']:
    do_anova(values_w, values_o, label, handle_nans=True)


#%% CORRELATIONS - GET BEHAVIOURAL DATA

## gotten from R - fitting logistic models
path = fname.behavioural_path + '/subject_slopes.csv'

def read_subject_slopes_csv(path, bad_subjects, subjects_with_no_BEM_simnibs):

    import csv
    with open(path, 'r') as csvfile:
        csv_file = csv.reader(csvfile)
        weak = list()
        omission = list()
        for line_index, line in enumerate(csv_file):
            if line_index == 0: ## header
                continue
            subject = line[2]
            if subject in bad_subjects or \
                subject in subjects_with_no_BEM_simnibs:
                weak.append(np.nan)
                omission.append(np.nan)
                continue
            weak.append(float(line[0]))
            omission.append(float(line[1]))

    return weak, omission        

weak, omission = read_subject_slopes_csv(path, bad_subjects,
                                         subjects_with_no_BEM_simnibs)

weak = np.array(weak)[~np.isnan(weak)]
omission = np.array(omission)[~np.isnan(omission)]


#%% CORRELATE WITH BEHAVIOUR

from scipy.stats import pearsonr
# plt.close('all')

def plot_correlation(test_array, behavioural_data, rho, p, typ, roi,
                     savepath, save=True,):
    
    mpl.rcParams.update(mpl.rcParamsDefault)
    mpl.rcParams['font.size'] = 14
    mpl.rcParams['font.weight'] = 'bold'
    
        
    
    fig = plt.figure()
    fig.set_size_inches(12, 9)
    plt.plot(behavioural_data, test_array, 'bo')
    plt.xlabel('Participant slopes (logistic regression)')
    plt.ylabel('Difference in envelope correlation (' + typ + \
               ', no-jitter minus jitter)')
    plt.title('Brain-behaviour correlation:\n' + typ + \
              ' envelope correlation: Cerebello-' + roi)
    # plt.show()
    
    #% LINEAR REGRESSION - even though the variables are not normally distributed
    
    X = np.array(behavioural_data)
    X = X[~np.isnan(X)]
    X = np.expand_dims(X, 1)
    X = np.concatenate((np.expand_dims(np.ones(len(X)).T, 1), X), axis=1)
    y = np.expand_dims(test_array, axis=1)
    
    beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
    
    def abline(slope, intercept):
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        plt.plot(x_vals, y_vals, 'b--')
        plt.show()
        
    abline(beta[1], beta[0])
    
    plt.legend(['Subjects', 'linear regression'])
    plt.text(-6.5, -0.015, 'Pearson correlation:\n' + \
             r'$\rho = $' + str(np.round(rho, 3)) + \
             '\n$\it{p}$ = ' + str(np.round(p, 4)),
             horizontalalignment='center')
    plt.show()
    
    if save:
        filename = typ + '_envelope_behaviour_correlation_' + roi + '.png'
        print('Saving: ' + filename)
        
        fig.savefig(join(savepath, filename), dpi=300)

def run_correlations(values_w, values_o, behavioural_data, roi, contrast):
    
    v = np.concatenate((np.expand_dims(values_w[roi]['no-jitter'], 0),
                        np.expand_dims(values_w[roi]['jitter'], 0),
                        np.expand_dims(values_o[roi]['no-jitter'], 0),
                        np.expand_dims(values_o[roi]['jitter'], 0)), axis=0)
    contrast = np.array(contrast)
    
    test_array = np.matmul(contrast, v)
    
    indices = ~np.isnan(test_array)
    
    r, p = pearsonr(test_array[indices], behavioural_data[indices])
    
    print('Rho: ' + str(r))
    print('p: ' + str(p))
    
    return test_array[indices], behavioural_data[indices], r, p
    
## weak
test_array, behavioural_data, rho, p = \
    run_correlations(values_w, values_o, weak, 'Thalamus_L', [1, -1, 0, 0])
plot_correlation(test_array, behavioural_data, rho, p, 'Weak', 'Thalamus_L',
                 fig_path)

test_array, behavioural_data, rho, p = \
    run_correlations(values_w, values_o, weak, 'Thalamus_R', [1, -1, 0, 0])
plot_correlation(test_array, behavioural_data, rho, p, 'Weak', 'Thalamus_R',
                 fig_path)


# ## omission
test_array, behavioural_data, rho, p = \
    run_correlations(values_w, values_o, omission, 'Thalamus_L', [0, 0, 1, -1])
plot_correlation(test_array, behavioural_data, rho, p, 'Omission',
                  'Thalamus_L', fig_path)
test_array, behavioural_data, rho, p = \
    run_correlations(values_w, values_o, omission, 'Thalamus_R', [0, 0, 1, -1])
plot_correlation(test_array, behavioural_data, rho, p, 'Omission',
                  'Thalamus_R', fig_path)

# ## interaction
# test_array, behavioural_data, rho, p = \
#     run_correlations(values_w, values_o, weak, 'Thalamus_R', [1, -1, -1, 1])
# plot_correlation(test_array, behavioural_data, rho, p, 'Weak', 'Thalamus_R')

# test_array, behavioural_data, rho, p = \
#     run_correlations(values_w, values_o, omission,
#                      'Thalamus_R', [1, -1, -1, 1])
# plot_correlation(test_array, behavioural_data, rho, p, 'Omission', 
#                  'Thalamus_R')

#%%

plt.close('all')
for roi in atlas['labels']:

    test_array, behavioural_data, rho, p = \
        run_correlations(values_w, values_o, weak, roi, [1, -1, 0, 0])
    plot_correlation(test_array, behavioural_data, rho, p, 'Weak', roi,
                     fig_path, save=False)


#%% A CIRCLE ATTEMPT


def plot_circle(ts, vertex_indices, seed):
    
    from mne_connectivity.viz import plot_connectivity_circle
    con = np.array(list(ts.values()))
    node_names = list(vertex_indices.keys())

    n_nodes = len(node_names)
    seed_index = node_names.index(seed)
    indices = (np.repeat(seed_index, n_nodes), np.arange(n_nodes))
    
    for node_name_index, node_name in enumerate(node_names):
        if node_name.rfind('Cerebelum') == 0:
            node_name = node_name.replace('Cerebelum', 'Cerebellum')
            node_names[node_name_index] = node_name
        
    fig = plt.figure(facecolor='black', figsize=(12, 9))
    plot_connectivity_circle(con, node_names, indices, vmin=-2, vmax=2,
                             colormap='bwr', fontsize_names=18,
                             fontsize_colorbar=26, fig=fig)#,
                             # title='Connectivity from Cerebellar Lobule VI')
    return fig

plt.close('all')
# plot_circle(ts_sub_R, vertex_indices, 'Cerebelum_6_R')
# fig = plot_circle(ts_w, all_vertex_indices, 'Cerebelum_6_L')
fig = plot_circle(ts_sub_w, vertex_indices, 'Cerebelum_6_L')
fig_path = '/home/lau/Nextcloud/arbejde/grants/ERC/2023/figures'
# fig.savefig(join(fig_path, 'conn_circle_pallidum.png'), dpi=300)
# fig.savefig(join(fig_path, 'conn_circle_thalamus.png'), dpi=300)
# plt.title('lala')

#%% second attempt

# def plot_circle_with_both(ts, vertex_indices, seeds):
    
#     from mne_connectivity.viz import plot_connectivity_circle
#     con = np.array(ts)
#     node_names = list(vertex_indices.keys())
#     n_nodes = len(node_names)
#     seed_indices = list()
#     for seed in seeds:
#         seed_indices.append(node_names.index(seed))
    
#     indices = (np.concatenate((np.repeat(seed_indices[0], n_nodes),
#                                np.repeat(seed_indices[1], n_nodes))),
#                np.concatenate((np.arange(n_nodes), np.arange(n_nodes))))

#     plot_connectivity_circle(con, node_names, indices, vmin=-2, vmax=2,
#                              colormap='seismic', n_lines=4)

# these_ts = np.array(list(ts_sub_L.values()) + list(ts_sub_R.values()))
#     # 
# plot_circle_with_both(these_ts, vertex_indices, ['Cerebelum_6_L',
#                                                  'Cerebelum_6_R'])

#%% CREATE MASKS

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


#%% PLOT MEDIAN STCS

def get_median_stc(a1, a2, seed, vertex_indices, src, ts_dict=None):

    n_subjects = len(a1)
    n_sources = a1.shape[1]
    subject_median_diff = np.zeros(shape=(n_subjects, n_sources))
    cutoff = abs(t.ppf(0.025, 24)) ## get df automatically
    
    
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
                    this_t = ts_dict[label]
                    if this_t > -cutoff and this_t < cutoff:
                        this_data[source_index] = 0
                    
        
    median_stc = mne.VolSourceEstimate(this_data, [src[0]['vertno']], 0, 1)
    
    return median_stc

def save_T1_plot_only(fig, filename):
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    axis_0 = fig.axes[0]
    extent = \
        axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, dpi=300, bbox_inches=extent.expanded(1.05, 1.17))


#%% PLOTS

# ts_dict = print_sig(theta_w0, theta_w15, 'Cerebelum_6_L', atlas
#                     , all_vertex_indices)
    
# median_stc = get_median_stc(theta_w0, theta_w15, 'Cerebelum_6_L',
#                             all_vertex_indices, src,
#                             ts_dict=ts_dict)
# median_stc.plot(src, colormap='bwr',
#                 clim=dict(kind='value', pos_lims=(0, 0.004, 0.008)))

# #########     
# median_stc = get_median_stc(theta_o0, theta_o15, 'Cerebelum_6_L',
#                             all_vertex_indices, src,
#                             ts_dict=print_sig(theta_o0, theta_o15,
#                                               'Cerebelum_6_L', atlas))
# median_stc.plot(src, colormap='bwr',
#                 clim=dict(kind='value', pos_lims=(0, 0.004, 0.008)))

########

mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams['font.size'] = 14
mpl.rcParams['font.weight'] = 'bold'
plt.close('all')

median_stc = get_median_stc(beta_w0, beta_w15, 'Cerebelum_6_L',
                            all_vertex_indices, src,
                            ts_dict=get_sig(beta_w0, beta_w15,
                                              'Cerebelum_6_L', atlas,
                                              all_vertex_indices))

# median_stc = get_median_stc(beta_w0, beta_w15, 'Cerebelum_6_L',
#                             all_vertex_indices, src,
#                             ts_dict=None)
fig = median_stc.plot(src, colormap='bwr',
                clim=dict(kind='value', pos_lims=(0, 0.004, 0.008)),
                initial_pos=(0.009, -0.023, 0.008))

save_T1_plot_only(fig, join(fig_path, 'envelope_beta_weak.png'))
########



median_stc = get_median_stc(beta_o0, beta_o15, 'Cerebelum_6_L',
                            all_vertex_indices, src,
                            ts_dict=get_sig(beta_o0, beta_o15,
                                              'Cerebelum_6_L', atlas,
                                              all_vertex_indices))
fig = median_stc.plot(src, colormap='bwr',
                clim=dict(kind='value', pos_lims=(0, 0.004, 0.008)),
                initial_pos=(0.009, -0.023, 0.008))
save_T1_plot_only(fig, join(fig_path, 'envelope_beta_omission.png'))


# #%% FIND NAN MEANS and DIFF

# #FIXME. find the label and summarize them at the subject stage

# mean_theta_w0 = np.nanmean(theta_w0, axis=0)
# mean_theta_w15 = np.nanmean(theta_w15, axis=0)

# mean_theta_o0 = np.nanmean(theta_o0, axis=0)
# mean_theta_o15 = np.nanmean(theta_o15, axis=0)

# mean_beta_w0 = np.nanmean(beta_w0, axis=0)
# mean_beta_w15 = np.nanmean(beta_w15, axis=0)

# mean_beta_o0 = np.nanmean(beta_o0, axis=0)
# mean_beta_o15 = np.nanmean(beta_o15, axis=0)


# #%% PLOT HIST

# plt.figure()
# plt.hist(theta_w0[0, :, :])
# plt.show()


# #%% PLOT DIFF

# plt.close('all')

# def get_diff(a1, a2, roi, vertex_indices, src, mask=True, median=True):
#     first =  np.mean(a1[:, vertex_indices[roi]], axis=1)
#     second = np.mean(a2[:, vertex_indices[roi]], axis=1)

#     first_stc  = mne.VolSourceEstimate(first, [src[0]['vertno']], 0, 1)
#     second_stc = mne.VolSourceEstimate(second, [src[0]['vertno']], 0, 1)

#     diff = first_stc.copy()
#     diff._data -= second_stc.data
    
#     if mask:
    
#         mask = np.zeros(n_sources)
#         for roi in vertex_indices:
#             mask[vertex_indices[roi]] = 1
    
#         mask = mask.astype(bool)
    
#         diff._data[~mask, :] = 0 ## mask
        
#     if median:
        
#         for roi in vertex_indices:
#             diff._data[vertex_indices[roi]] = \
#                 np.median(diff.data[vertex_indices[roi]])
        
#     return diff

# def save_T1_plot_only(fig, filename):
#     mpl.rcParams['font.size'] = 8

#     fig_manager = plt.get_current_fig_manager()
#     fig_manager.window.showMaximized()
#     axis_0 = fig.axes[0]
#     extent = \
#         axis_0.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
#     cbar_text = axis_0.get_children()[-2]
    
    
#     if 'theta_weak' in filename:
#         string = \
#             'Envelope correlation, Left SI, 4-7 Hz, difference: Weak, non-jittered minus jittered'
#     if 'theta_omission' in filename:
#         string = \
#             'Envelope correlation, Left SI, 4-7 Hz, difference: Omission, non-jittered minus jittered'
#     if 'beta_weak' in filename:
#         string = \
#             'Envelope correlation, Left SI, 14-30 Hz, difference: Weak, non-jittered minus jittered'
#     if 'beta_omission' in filename:
#         string = \
#             'Envelope correlation, Left SI, 14-30 Hz, difference: Omission, non-jittered minus jittered'
#     cbar_text.set_text(string)
#     cbar_text.set_fontsize('small')
#     fig.savefig(filename, dpi=300, bbox_inches=extent.expanded(1.05, 1.17))



# save_path = fname.subject_figure_path(subject='fsaverage',
#                                       date='20210825_000000')



# # diff_theta_w = get_diff(mean_theta_w0, mean_theta_o15, 'Postcentral_L', 
# #                         vertex_indices, src,
# #                         median=True, mask=True)

# # diff_theta_o = get_diff(mean_theta_o0, mean_theta_o15, 'Postcentral_L', 
# #                         vertex_indices, src,
# #                         median=True, mask=True)

# diff_beta_w = get_diff(mean_beta_w0, mean_beta_w15, 'Postcentral_L',
#                         vertex_indices, src,
#                         median=True, mask=True)

# diff_beta_o = get_diff(mean_beta_o0, mean_beta_o15, 'Postcentral_L', 
#                         vertex_indices, src,
#                         median=True, mask=True)




# # fig = diff_theta_w.plot(src, colormap='bwr',
# #             clim=dict(kind='value', pos_lims=(0.000, 0.001, 0.002)),
# #             initial_pos=(-0.030, -0.007, -0.007))
# #                        # pos_lims=(np.quantile(diff_theta_w.data, 0.90),
# #                        #           np.quantile(diff_theta_w.data, 0.95),
# #                        #           np.quantile(diff_theta_w.data, 0.975))))
                       
# # save_T1_plot_only(fig, filename=join(save_path, 'envelope_theta_weak.png'))   
 

# # fig = diff_theta_o.plot(src, colormap='bwr',
# #             clim=dict(kind='value', pos_lims=(0.000, 0.001, 0.002)),
# #             initial_pos=(-0.030, -0.007, -0.007))

# # save_T1_plot_only(fig, filename=join(save_path, 'envelope_theta_omission.png'))    



# # fig = diff_beta_w.plot(src, colormap='bwr',
# #             clim=dict(kind='value', pos_lims=(0.0000, 0.001, 0.002)),
# #             initial_pos=(-0.030, -0.007, -0.007))

# # save_T1_plot_only(fig, filename=join(save_path, 'envelope_beta_weak.png'))    


# fig = diff_beta_o.plot(src, colormap='bwr',
#             clim=dict(kind='value', pos_lims=(0.0000, 0.001, 0.002)),
#             initial_pos=(-0.030, -0.007, -0.007))
#                        # pos_lims=(np.quantile(diff_beta_o.data, 0.90),
#                        #          np.quantile(diff_beta_o.data, 0.95),
#                        #          np.quantile(diff_beta_o.data, 0.975))))

# # save_T1_plot_only(fig, filename=join(save_path, 'envelope_beta_omission.png'))



