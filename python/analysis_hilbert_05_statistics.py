#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:20:54 2022

@author: lau
"""

from config import (fname, submitting_method,
                    hilbert_stat_contrasts,
                    hilbert_tmin, hilbert_tmax,
                    hilbert_fmins, hilbert_fmaxs,
                    hilbert_stat_tmin, hilbert_stat_tmax,
                    hilbert_stat_p, hilbert_stat_n_permutations,
                    hilbert_stat_n_jobs, hilbert_stat_seed,
                    hilbert_stat_channels,
                    recordings, bad_subjects, bad_channels)
from sys import argv
from helper_functions import should_we_run, find_common_channels

import mne
import numpy as np
from scipy import stats

def this_function(subject, date, overwrite):
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        for this_contrast in hilbert_stat_contrasts:
            print(this_contrast)
            output_name = fname.hilbert_statistics_proj(
                subject=subject, date=date, fmin=fmin, fmax=fmax,
                tmin=hilbert_tmin, tmax=hilbert_tmax,
                first_event=this_contrast[0],
                second_event=this_contrast[1],
                stat_tmin=hilbert_stat_tmin, stat_tmax=hilbert_stat_tmax,
                nperm=hilbert_stat_n_permutations, seed=hilbert_stat_seed,
                pval=hilbert_stat_p)
        
            if should_we_run(output_name, overwrite):
                subject_counter = 0
                for recording_index, recording in enumerate(recordings):
                    subject_name = recording['subject']
                    if subject_name in bad_subjects:
                        continue # skip the subject
                    print('Loading subject: ' + subject_name)
                    subject_counter += 1
                    date = recording['date']
                    
                    wilcoxons = mne.read_evokeds(
                        fname.hilbert_wilcoxon_proj(
                                    subject=subject_name, date=date,
                                    fmin=fmin, fmax=fmax,
                                    tmin=hilbert_tmin, tmax=hilbert_tmax))
                    
                    for wilcoxon in wilcoxons:
                        if this_contrast[0] in wilcoxon.comment \
                                and this_contrast[1] in wilcoxon.comment:
                            break ## find the right contrast
                    channels = np.setdiff1d(wilcoxon.info['ch_names'],
                                            find_common_channels(bad_channels))
                    wilcoxon.pick_channels(channels)
                    wilcoxon.pick_types(hilbert_stat_channels)
                    wilcoxon.crop(hilbert_stat_tmin, hilbert_stat_tmax)
                    
                    if recording_index == 0:
                        data_array = wilcoxon.data.T
                        stat_array = np.expand_dims(data_array, axis=0)
                    else:
                        this_data = wilcoxon.data.T
                        this_data = np.expand_dims(this_data, axis=0)
                        stat_array = np.concatenate((stat_array,
                                                     this_data),
                                                    axis=0)
        
                print(stat_array.shape)
                n_subjects = stat_array.shape[0]
                
                grand_average = mne.read_evokeds(
                    fname.hilbert_grand_average_proj(subject=subject,
                                                     date=date,
                                                     fmin=fmin,
                                                     fmax=fmax,
                                                     tmin=hilbert_tmin,
                                                     tmax=hilbert_tmax))[0]
                
                ch_adjacency, ch_names = \
                    mne.channels.find_ch_adjacency(grand_average.info,
                                               ch_type=hilbert_stat_channels)                
                

                
                threshold = -stats.distributions.t.ppf(hilbert_stat_p / 2.0,
                                                       n_subjects)
                print('Running cluster statistics for: ' + str(this_contrast))
                t_obs, clusters, cluster_p_values, H0 = clu = \
                    mne.stats.spatio_temporal_cluster_1samp_test(
                        stat_array, threshold,
                        n_permutations=hilbert_stat_n_permutations,
                        n_jobs=hilbert_stat_n_jobs,
                        seed=hilbert_stat_seed,
                        adjacency=ch_adjacency)
                    
                cluster_dict = dict()
                names = ['t_obs', 'clusters', 'cluster_p_values', 'H0']
                for name_index, name in enumerate(names):
                    cluster_dict[name] = clu[name_index]
                np.save(output_name, cluster_dict)
    
if submitting_method == 'hyades_frontend':
    queue = 'long.q'
    job_name = 'hstat'
    n_jobs = hilbert_stat_n_jobs
    deps = ['eve', 'hfilt', 'hepo', 'have', 'hwil']

    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))