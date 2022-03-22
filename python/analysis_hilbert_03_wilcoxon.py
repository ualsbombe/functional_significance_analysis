#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 11:31:29 2021

@author: lau
"""

from config import (fname, submitting_method, hilbert_fmins, hilbert_fmaxs,
                    hilbert_tmin, hilbert_tmax, hilbert_contrasts,
                    collapsed_event_id)
from sys import argv
from helper_functions import should_we_run, wilcoxon, collapse_event_id
import mne
import numpy as np

def this_function(subject, date, overwrite):
    
    for hilbert_fmin, hilbert_fmax in zip(hilbert_fmins, hilbert_fmaxs):
        z_transformed_proj = list()        
        z_transformed_no_proj = list()
        
        for contrast in hilbert_contrasts:
            contrast_name = contrast[0] + '_vs_' + contrast[1]
            print('Running contrast: ' + contrast_name)
        
            output_names = list()
            # output_names.append(fname.hilbert_wilcoxon_no_proj(
            #                             subject=subject,
            #                             date=date,
            #                             fmin=hilbert_fmin,
            #                             fmax=hilbert_fmax,
            #                             tmin=hilbert_tmin,
            #                             tmax=hilbert_tmax))
            output_names.append(fname.hilbert_wilcoxon_proj(
                                        subject=subject,
                                        date=date,
                                        fmin=hilbert_fmin,
                                        fmax=hilbert_fmax,
                                        tmin=hilbert_tmin,
                                        tmax=hilbert_tmax))
            
            for output_name in output_names:
                if should_we_run(output_name, overwrite):

                    if 'no_proj' in output_name:
                        proj = False
                        input_name = fname.hilbert_epochs_no_proj(
                                    subject=subject,
                                    date=date,
                                    fmin=hilbert_fmin,
                                    fmax=hilbert_fmax,
                                    tmin=hilbert_tmin,
                                    tmax=hilbert_tmax)
                    else:
                        proj = True
                        input_name = fname.hilbert_epochs_proj(
                                    subject=subject,
                                    date=date,
                                    fmin=hilbert_fmin,
                                    fmax=hilbert_fmax,
                                    tmin=hilbert_tmin,
                                    tmax=hilbert_tmax)
                    
                    epochs = mne.read_epochs(input_name, proj=proj,
                                             preload=False)
                    if not proj:
                        epochs.del_proj()
                        
                    if contrast_name == 'w0_vs_w15' or \
                                                contrast_name == 'o0_vs_o15':
                        epochs = collapse_event_id(epochs, collapsed_event_id)
                        
                    first_event  = epochs[contrast[0]]
                    second_event = epochs[contrast[1]]
                    first_event.load_data()
                    second_event.load_data()
                    first_event.pick_types(meg=True)
                    second_event.pick_types(meg=True)
                    mne.epochs.equalize_epoch_counts([first_event,
                                                      second_event])
                    data_1 = first_event.get_data()
                    data_2 = second_event.get_data()
                    
                    n_channels, n_samples = data_1.shape[1:]
                    zs = np.zeros((n_channels, n_samples))
                    
                    for channel_index in range(n_channels):
                        for sample_index in range(n_samples):
                            this_data_1 = data_1[:, channel_index,
                                                 sample_index]
                            this_data_2 = data_2[:, channel_index,
                                                 sample_index]
                            z = wilcoxon(abs(this_data_1), abs(this_data_2))
                            zs[channel_index, sample_index] = z
                    z_transform = mne.EvokedArray(zs, info=first_event.info,
                                                  tmin=first_event.tmin,
                                                  comment=contrast_name,
                                                  nave=this_data_1.shape[0])
                    if proj:
                        z_transformed_proj.append(z_transform)
                    else:
                        z_transformed_no_proj.append(z_transform)
        # mne.write_evokeds(output_names[0], z_transformed_no_proj)
        mne.write_evokeds(output_names[0], z_transformed_proj)
    epochs = None # memory control
                    
                    
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'hwil'
    n_jobs = 3
    deps = ['eve', 'hfilt', 'hepo', 'have']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))             