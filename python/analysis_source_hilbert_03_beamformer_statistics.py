#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:43:35 2022

@author: lau
"""

from config import (fname, submitting_method, src_spacing,
                    hilbert_lcmv_stat_contrasts,
                    hilbert_lcmv_regularization,
                    hilbert_tmin, hilbert_tmax,
                    hilbert_fmins, hilbert_fmaxs,
                    hilbert_lcmv_stat_tmin, hilbert_lcmv_stat_tmax,
                    hilbert_lcmv_stat_p, hilbert_lcmv_stat_n_permutations,
                    hilbert_lcmv_stat_n_jobs, hilbert_lcmv_stat_seed,
                    hilbert_lcmv_stat_connectivity_dist,
                    recordings, bad_subjects, subjects_with_no_BEM_simnibs)
from sys import argv
from helper_functions import should_we_run

import mne
import numpy as np
from scipy import stats

def this_function(subject, date, overwrite):
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        for this_contrast in hilbert_lcmv_stat_contrasts:
            print(this_contrast)
            output_name = fname.source_hilbert_beamformer_statistics(
                subject=subject, date=date, fmin=fmin, fmax=fmax,
                tmin=hilbert_tmin, tmax=hilbert_tmax,
                reg=hilbert_lcmv_regularization,
                first_event=this_contrast[0],
                second_event=this_contrast[1],
                stat_tmin=hilbert_lcmv_stat_tmin,
                stat_tmax=hilbert_lcmv_stat_tmax,
                nperm=hilbert_lcmv_stat_n_permutations,
                seed=hilbert_lcmv_stat_seed,
                condist=hilbert_lcmv_stat_connectivity_dist,
                pval=hilbert_lcmv_stat_p)
        
            if should_we_run(output_name, overwrite):
                subject_counter = 0
                for recording_index, recording in enumerate(recordings):
                    subject_name = recording['subject']
                    if subject_name in bad_subjects or \
                        subject_name in subjects_with_no_BEM_simnibs:
                        continue # skip the subject
                    print('Loading subject: ' + subject_name)
                    subject_counter += 1
                    date = recording['date']
                    
                    lcmv = mne.read_source_estimate(
                        fname.source_hilbert_beamformer_contrast_simnibs_morph(
                            subject=subject_name,date=date,
                            fmin=fmin, fmax=fmax,
                            tmin=hilbert_tmin,
                            tmax=hilbert_tmax,
                            reg=hilbert_lcmv_regularization,
                            first_event=this_contrast[0],
                            second_event=this_contrast[1],
                            n_layers=3,
                        weight_norm='unit-noise-gain-invariant'))

                    lcmv.crop(hilbert_lcmv_stat_tmin,
                              hilbert_lcmv_stat_tmax)

                    
                    if recording_index == 0:
                        data_array = lcmv.data.T
                        stat_array = np.expand_dims(data_array, axis=0)
                    else:
                        this_data = lcmv.data.T
                        this_data = np.expand_dims(this_data, axis=0)
                        stat_array = np.concatenate((stat_array,
                                                     this_data),
                                                    axis=0)

                  
          
            n_subjects = stat_array.shape[0]
            n_times = stat_array.shape[1]
            
            src = mne.source_space.read_source_spaces(
                fname.anatomy_volumetric_source_space(subject='fsaverage',
                                                      spacing=src_spacing))
            
            connectivity = mne.spatio_temporal_src_adjacency(src, n_times,
                                          hilbert_lcmv_stat_connectivity_dist)
            
            # threshold = -stats.distributions.t.ppf(hilbert_lcmv_stat_p / 2.0,
            #                                        n_subjects)
            threshold = None
            print('Running cluster statistics for: ' + str(this_contrast))
            t_obs, clusters, cluster_p_values, H0 = clu = \
                mne.stats.spatio_temporal_cluster_1samp_test(
                    stat_array, threshold,
                    n_permutations=hilbert_lcmv_stat_n_permutations,
                    n_jobs=hilbert_lcmv_stat_n_jobs,
                    seed=hilbert_lcmv_stat_seed,
                    adjacency=connectivity)
                
            cluster_dict = dict()
            names = ['t_obs', 'clusters', 'cluster_p_values', 'H0']
            for name_index, name in enumerate(names):
                cluster_dict[name] = clu[name_index]
            np.save(output_name, cluster_dict)
        
if submitting_method == 'hyades_frontend':
    queue = 'highmem_short.q'
    job_name = 'hstatlcmv'
    n_jobs = hilbert_lcmv_stat_n_jobs
    deps = ['eve', 'hfilt', 'hepo', 'have', 'mri', 'ana', 'fwd', 'hlcmv',
            'mhlcmv']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))