#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:08:42 2022

@author: lau
"""

from config import (fname, submitting_method, src_spacing,
                    evoked_lcmv_stat_contrasts,
                    evoked_lcmv_regularization,
                    evoked_tmin, evoked_tmax,
                    evoked_fmin, evoked_fmax,
                    evoked_lcmv_stat_tmin, evoked_lcmv_stat_tmax,
                    evoked_lcmv_stat_p, evoked_lcmv_stat_n_permutations,
                    evoked_lcmv_stat_n_jobs, evoked_lcmv_stat_seed,
                    evoked_lcmv_stat_connectivity_dist,
                    recordings, bad_subjects)
from sys import argv
from helper_functions import should_we_run

import mne
import numpy as np
from scipy import stats

def this_function(subject, date, overwrite):
    for this_contrast in evoked_lcmv_stat_contrasts:
        print(this_contrast)
        data_arrays = list()
        for event in this_contrast:
            output_name = fname.source_evoked_beamformer_statistics(
                subject=subject, date=date, fmin=evoked_fmin, fmax=evoked_fmax,
                tmin=evoked_tmin, tmax=evoked_tmax,
                reg=evoked_lcmv_regularization,
                event=event,
                first_event=this_contrast[0], second_event=this_contrast[1],
                stat_tmin=evoked_lcmv_stat_tmin,
                stat_tmax=evoked_lcmv_stat_tmax,
                nperm=evoked_lcmv_stat_n_permutations,
                seed=evoked_lcmv_stat_seed,
                condist=evoked_lcmv_stat_connectivity_dist,
                pval=evoked_lcmv_stat_p)
        
            if should_we_run(output_name, overwrite):
                subject_counter = 0
                for recording_index, recording in enumerate(recordings):
                    subject_name = recording['subject']
                    if subject_name in bad_subjects:
                        continue # skip the subject
                    print('Loading subject: ' + subject_name)
                    subject_counter += 1
                    date = recording['date']
                    
                    lcmv = mne.read_source_estimate(
                        fname.source_evoked_beamformer_morph(
                            subject=subject_name,date=date,
                            fmin=evoked_fmin, fmax=evoked_fmax,
                            tmin=evoked_tmin,
                            tmax=evoked_tmax,
                            reg=evoked_lcmv_regularization, event=event,
                            first_event=this_contrast[0],
                            second_event=this_contrast[1]) + '-stc.h5')
                    
                    lcmv.crop(evoked_lcmv_stat_tmin, evoked_lcmv_stat_tmax)
                    
                    if recording_index == 0:
                        data_array = lcmv.data.T
                        data_array = np.expand_dims(data_array, axis=0)
                    else:
                        this_data = lcmv.data.T
                        this_data = np.expand_dims(this_data, axis=0)
                        data_array = np.concatenate((data_array, this_data),
                                                    axis=0)
                data_arrays.append(data_array)
        
        stat_array = data_arrays[0] - data_arrays[1]
        n_subjects = stat_array.shape[0]
        n_times = stat_array.shape[1]
        
        src = mne.source_space.read_source_spaces(
            fname.anatomy_volumetric_source_space(subject='fsaverage',
                                                  spacing=src_spacing))
        
        connectivity = mne.spatio_temporal_src_adjacency(src, n_times,
                                      evoked_lcmv_stat_connectivity_dist)
        
        threshold = -stats.distributions.t.ppf(evoked_lcmv_stat_p / 2.0,
                                               n_subjects)
        print('Running cluster statistics for: ' + str(this_contrast))
        t_obs, clusters, cluster_p_values, H0 = clu = \
            mne.stats.spatio_temporal_cluster_1samp_test(
                stat_array, threshold,
                n_permutations=evoked_lcmv_stat_n_permutations,
                n_jobs=evoked_lcmv_stat_n_jobs,
                seed=evoked_lcmv_stat_seed,
                adjacency=connectivity)
            
        cluster_dict = dict()
        names = ['t_obs', 'clusters', 'cluster_p_values', 'H0']
        for name_index, name in enumerate(names):
            cluster_dict[name] = clu[name_index]
        np.save(output_name, cluster_dict)
        
if submitting_method == 'hyades_frontend':
    queue = 'all.q'
    job_name = 'estatlcmv'
    n_jobs = evoked_lcmv_stat_n_jobs
    deps = ['eve', 'efilt', 'eepo', 'eave', 'mri', 'ana', 'fwd', 'elcmv',
            'emlcmv']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))