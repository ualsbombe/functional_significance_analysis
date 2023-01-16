#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 12 10:06:30 2021

@author: lau
"""

#%% IMPORTS AND SUBJECT

from config import recordings, fname
import mne
import mne_connectivity
import numpy as np
import matplotlib.pyplot as plt

subject = recordings[0]['subject']
date = recordings[0]['date']

#%% GET EPOCHS

epochs_filename = fname.hilbert_epochs_no_proj(subject=subject, date=date,
                                           fmin=4, fmax=7,
                                           tmin=-0.750, tmax=0.750)
    
epochs = mne.read_epochs(epochs_filename, preload=False, proj=True)
# epochs.del_proj()

event_names = ['w0_hit' , 'w0_miss',
          'w15_hit', 'w15_miss' ]

def concatenate(epochs, event_names):
    new_epochs = [None] * len(event_names)
    for event_name_index, event_name in enumerate(event_names):
        new_epochs[event_name_index] = epochs[event_name]
    
    new_epochs = mne.epochs.concatenate_epochs(new_epochs)
    return new_epochs
    
epochs = concatenate(epochs, event_names)

epochs.pick_types(meg='grad')
# epochs.crop(-0.300, 0.300)

#%% CREATE EPOCHS COV
raw_filename = fname.raw_file(subject=subject, date=date)
raw = mne.io.read_raw_fif(raw_filename, preload=True)
raw.filter(4, 7)

events = mne.read_events(fname.events(subject=subject, date=date))

epochs_cov = mne.Epochs(raw, events, epochs.event_id, tmin=-0.750, tmax=0.750,
                        baseline=None, proj=True, preload=True)

del raw

#%% FORWARD MODEL

trans = fname.anatomy_transformation(subject=subject,
                                     date=date)
src = fname.anatomy_volumetric_source_space(
    subject=subject, spacing=7.5)
bem = fname.anatomy_simnibs_bem_solutions(
    subject=subject, n_layers=3)

fwd = mne.make_forward_solution(epochs_cov.info, trans,
                                src, bem)


#%% CREATE SPATIAL FILTER


# epochs_cov.del_proj()
epochs_cov.pick_types(meg='grad')

data_cov = mne.compute_covariance(epochs_cov)#, rank=dict(mag=60), tmin=-0.500, tmax=0.500)

filters = mne.beamformer.make_lcmv(epochs.info, fwd, data_cov, reg=0.05,
                                   pick_ori='max-power')

evoked = epochs.average()

#%% RECONSTRUCT EPOCHS

stcs = mne.beamformer.apply_lcmv_epochs(epochs[:10], filters, return_generator=False)
stc_evoked = mne.beamformer.apply_lcmv(evoked, filters)

#%% MORPH

# morph = mne.read_source_morph(
#     fname.anatomy_simnibs_morph_volume(subject=subject, spacing=7.5))
# morph.compute_vol_morph_mat()
# mat = morph.vol_morph_mat
# src_fsaverage = mne.read_source_spaces(
#     fname.anatomy_volumetric_source_space(subject='fsaverage', spacing=7.5))
#%% DOWNSAMPLE 


# stcs_morph = [None] * len(stcs)

for stc_index, stc in enumerate(stcs):
    # stc.resample(100)
    stc.crop(-0.100, 0.100)
    # stcs_morph[stc_index] = morph.apply(stc)
    
#%% CONNECTIVITY

corr = mne_connectivity.envelope_correlation(stcs, verbose=True)
# corr_morph = mne_connectivity.envelope_correlation(stcs_morph, verbose=True)


#%% PLOT MAT

this_data = corr.get_data().mean(axis=0).squeeze()

plt.figure()
plt.imshow(this_data)
plt.show()

plt.figure()
plt.hist(this_data)
plt.show()


# #%% DEGREE?

# corr_data = np.mean(corr.get_data(), axis=0)[:, :, 0]
# corr_morph_data = corr_morph.get_data().mean(axis=0).squeeze()
# corr_morph_data[np.isnan(corr_morph_data)] = -1 # is this okay?
# corr_morph_data[np.isnan(corr_morph_data)] = 10 # is this okay?


# # max vertex = 2490
# # in the cerebellum somewhere 245
# max_conn = corr_data[245, :]

# degree = mne_connectivity.degree(corr_data, 0.1)
# degree_morph = mat @ degree

# stc_degree = mne.VolSourceEstimate(degree, [fwd['src'][0]['vertno']], 0, 1)
# stc_degree_morph = mne.VolSourceEstimate(degree_morph,
#                                          [src_fsaverage[0]['vertno']], 0, 1)

# stc_corr = mne.VolSourceEstimate(corr_data, [fwd['src'][0]['vertno']], 0, 1)
# stc_corr_morph = mne.VolSourceEstimate(corr_morph_data, [src_fsaverage[0]['vertno']], 
#                                 0, 1)

# ## Sune's advice

# sune_morph = mat @ corr_data * mat.T
# stc_sune_morph = mne.VolSourceEstimate(sune_morph, [src_fsaverage[0]['vertno']], 
#                                 0, 1)
# #%% PLOT

# stc_degree.plot(fwd['src'])
# stc_degree_morph.plot(src_fsaverage)
# stc_corr.plot(fwd['src'], initial_pos=(-0.0, -0.0, 0.0))#,
#               # clim=dict(kind='value', lims=(0.00, 0.25, 0.50)))
# stc_corr_morph.plot(src_fsaverage, initial_pos=(-0.0, -0.0, 0.0))#,
#                # clim=dict(kind='value', lims=(0.00, 0.25, 0.50)))
# stc_sune_morph.plot(src_fsaverage, initial_pos=(0, 0, 0))              

# #%% MORPH TO FSAVERAGE

# data_morph = np.mean(corr_morph.get_data(), axis=0).squeeze()
# stc_max_morph = mne.VolSourceEstimate(data_morph, [src_fsaverage[0]['vertno']], 0, 1)

# #%% PLOT MORPH

# stc_max_morph.plot(src_fsaverage, clim=dict(kind='value', lims=(0, 0.2, 0.3)))

# #%% PLOT

# import matplotlib.pyplot as plt

# plt.close('all')

# plt.figure()
# plt.spy(mat)
# plt.show()

# plt.figure()
# plt.plot(mat.data)
# plt.show()

# plt.figure()
# plt.imshow(corr_data)
# plt.show()

# stc_from = stc_evoked
# stc_evoked_morph = morph.apply(stc_evoked)

# ## test
# data_test = mat @ stc_from.data
# # data_test - stc_evoked_morph == 0

# #%% playing around

# stc_corr = mne.VolSourceEstimate(corr_data[:, :], [fwd['src'][0]['vertno']], 0, 1)
# stc_corr.plot(fwd['src'], clim=dict(kind='value', lims=(0.40, 0.45, 0.50)),
#               mode='glass_brain')

# #%% playing around

# lala  = mat @ corr_data
# lala_T = mat @ corr_data.T

# plt.close('all')

# plt.figure()
# plt.imshow(lala)
# plt.title('Normal')
# plt.show()


# plt.figure()
# plt.imshow(lala_T)
# plt.title('Transposed')
# plt.show()

# super_lala = lala @ lala.T

# plt.figure()
# plt.imshow(super_lala)
# plt.show()

# super_lala_norm = (super_lala - np.min(super_lala)) / \
#                   (np.max(super_lala) - np.min(super_lala))
                  
# plt.figure()
# plt.imshow(super_lala_norm)
# plt.show()

# stc_corr_lala = mne.VolSourceEstimate(lala[:, 245],
#                                        [src_fsaverage[0]['vertno']], 0, 1)

# stc_corr_lala.plot(src_fsaverage)


# stc_corr_morph = mne.VolSourceEstimate(super_lala_norm,
#                                        [src_fsaverage[0]['vertno']], 0, 1)

# stc_corr_morph.plot(src_fsaverage)

# stc_corr_morph_roi = mne.VolSourceEstimate(super_lala_norm[:, 4300],
#                                            [src_fsaverage[0]['vertno']], 0, 1)

# stc_corr_morph_roi.plot(src_fsaverage)

# #%% toy version

# toy_corr_data = corr_data[:4068, :3]
# toy_mat = mat[:3, :]

# toy_lala = toy_mat @ toy_corr_data

# #%% Sune's advice