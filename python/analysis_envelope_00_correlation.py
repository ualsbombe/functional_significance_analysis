#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 09:02:57 2022

@author: lau
"""

from config import (fname, submitting_method, envelope_events,
                    envelope_fmins, envelope_fmaxs, hilbert_tmin, hilbert_tmax,
                    envelope_tmin, envelope_tmax, envelope_regularization,
                    envelope_weight_norm, envelope_picks, bad_channels,
                    collapsed_event_id, subjects_conn_cannot_be_saved,
                    src_spacing, bem_conductivities)
from sys import argv
from helper_functions import should_we_run, collapse_event_id

import mne
import mne_connectivity
import numpy as np

def this_function(subject, date, overwrite):
    bem_conductivity = bem_conductivities[1]
    n_layers = len(bem_conductivity)
    morph = mne.read_source_morph(fname.anatomy_simnibs_morph_volume(
        subject=subject, spacing=src_spacing))
    for (fmin, fmax) in zip(envelope_fmins, envelope_fmaxs):
        raw_loaded = False
        ## FIXME: second pair of events are not run
        for these_events in envelope_events:
            output_names = [None] * 2
            output_names_np = [None] * 2
            output_names[0] = fname.envelope_correlation(
                subject=subject, date=date, fmin=fmin, fmax=fmax,
                tmin=envelope_tmin, tmax=envelope_tmax,
                reg=envelope_regularization, weight_norm=envelope_weight_norm,
                n_layers=n_layers, event=these_events[0],
                first_event=these_events[0], second_event=these_events[1])
            output_names[1] = fname.envelope_correlation(
                subject=subject, date=date, fmin=fmin, fmax=fmax,
                tmin=envelope_tmin, tmax=envelope_tmax,
                reg=envelope_regularization, weight_norm=envelope_weight_norm,
                n_layers=n_layers, event=these_events[1],
                first_event=these_events[0], second_event=these_events[1])
            
            output_names_np[0] = \
                                fname.envelope_correlation_morph_data(
                                    subject=subject, date=date,
                                    fmin=fmin, fmax=fmax,
                                    tmin=envelope_tmin, tmax=envelope_tmax,
                                    reg=envelope_regularization,
                                    event=these_events[0],
                                    first_event=these_events[0],
                                    second_event=these_events[1],
                                    n_layers=n_layers,
                                    weight_norm=envelope_weight_norm)
            output_names_np[1] = \
                                fname.envelope_correlation_morph_data(
                                    subject=subject, date=date,
                                    fmin=fmin, fmax=fmax,
                                    tmin=envelope_tmin, tmax=envelope_tmax,
                                    reg=envelope_regularization,
                                    event=these_events[1],
                                    first_event=these_events[0],
                                    second_event=these_events[1],
                                    n_layers=n_layers,
                                    weight_norm=envelope_weight_norm)

                        
            if should_we_run(output_names[0], overwrite) or \
                should_we_run(output_names[1], overwrite) or \
                should_we_run(output_names_np[0], overwrite) or \
                should_we_run(output_names_np[1], overwrite):
                if morph.vol_morph_mat is None: # only compute once
                    morph.compute_vol_morph_mat()

                if not raw_loaded:
                    raw = mne.io.read_raw_fif(fname.hilbert_filter(
                        subject=subject, date=date, fmin=fmin, fmax=fmax),
                        preload=False)
                    # raw_loaded = True
                    epochs_hilbert = \
                        mne.read_epochs(fname.hilbert_epochs_no_proj(
                            subject=subject, date=date, fmin=fmin,
                            fmax=fmax,
                            tmin=hilbert_tmin, tmax=hilbert_tmax),
                            proj=False, preload=False)
                        
                    ## apply bads
                    raw.info['bads'] = bad_channels[subject]
                    epochs_hilbert.info['bads'] = bad_channels[subject]
                    
                    picks = mne.pick_types(epochs_hilbert.info,
                                           meg=envelope_picks)
                    
                    epochs_hilbert = collapse_event_id(epochs_hilbert,
                                                      collapsed_event_id)
                    
                    events = epochs_hilbert.events
                    baseline = epochs_hilbert.baseline
                    event_ids = epochs_hilbert.event_id
                    
                    ## only look at chosen event pair
                    new_event_id = dict()
                    for event in event_ids:
                        if event in these_events:
                            new_event_id[event] = event_ids[event]
                            
                    epochs_cov = mne.Epochs(raw, events, new_event_id,
                                            hilbert_tmin, hilbert_tmax,
                                            baseline, proj=False,
                                            preload=True, picks=picks)
                    
                    ## remove projs
                    epochs_hilbert.del_proj()
                    epochs_cov.del_proj()
                    rank = None
                    
                    ## make forward model on the fly
                    
                    trans = fname.anatomy_transformation(subject=subject,
                                                         date=date)
                    src = fname.anatomy_volumetric_source_space(
                        subject=subject, spacing=src_spacing)
                    bem = fname.anatomy_simnibs_bem_solutions(
                        subject=subject, n_layers=n_layers)
                    
                    fwd = mne.make_forward_solution(epochs_cov.info, trans,
                                                    src, bem)
                    ## estimate spatial filter
                    data_cov = mne.compute_covariance(epochs_cov,
                                                      tmin=None, tmax=None,
                                                      rank=rank)
                    
                    filters = mne.beamformer.make_lcmv(
                        epochs_cov.info, fwd, data_cov,
                        pick_ori='max-power',
                        weight_norm=envelope_weight_norm,
                        reg=envelope_regularization)
                    
                    del epochs_cov ## release memory
                    
                    print('Reconstructing events:' + str(these_events))
                    
                    for event_index, event in enumerate(these_events):
                        print('Reconstructing event: ' + event)
                        these_epochs = epochs_hilbert[event]
                        these_epochs.load_data()
                        these_epochs.pick(picks)
                        
                        stcs = mne.beamformer.apply_lcmv_epochs(
                            these_epochs, filters, return_generator=False)
                        
                        ## downsample
                        stcs_morph = [None] * len(stcs)
                        for stc_index, stc in enumerate(stcs):
                            stc.crop(envelope_tmin, envelope_tmax)
                            stcs_morph[stc_index] = morph.apply(stc)
                    
                        ## envelope correlations
                        print('Running envelope correlations')
                        if subject not in subjects_conn_cannot_be_saved:
                            corr = mne_connectivity.envelope_correlation(
                                stcs, verbose=True)
                        corr_morph = mne_connectivity.envelope_correlation(
                            stcs_morph, verbose=True)
                        
                        corr_morph_data = \
                            corr_morph.get_data().mean(axis=0).squeeze()
                        # is this okay?
                        # corr_morph_data[np.isnan(corr_morph_data)] = -1 

                        if subject not in subjects_conn_cannot_be_saved:
                            print('Saving: ' + output_names[event_index])
                            corr.save(output_names[event_index])
                            ## add a numpy saving instead??!
                            
                        
                        # np.output
                        output_name_np = output_names_np[event_index]
                        print('Saving: ' + output_name_np)
                        np.save(output_name_np, corr_morph_data)

                                

                            
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'envco'
    n_jobs = 3
    deps = ['eve', 'hfilt', 'hepo', 'mri', 'ana', 'fwd', 'snibs', 'snbem']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))
                      
                    