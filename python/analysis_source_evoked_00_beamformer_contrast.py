#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 11:28:54 2021

@author: lau
"""

from config import (fname, submitting_method, evoked_lcmv_contrasts,
                    evoked_lcmv_weight_norms, evoked_lcmv_regularization,
                    evoked_lcmv_picks, evoked_lcmv_proj,
                    evoked_tmin, evoked_tmax,
                    evoked_fmin, evoked_fmax,
                    bad_channels, collapsed_event_id, src_spacing, bem_ico,
                    bem_conductivities)
from sys import argv
from helper_functions import should_we_run, collapse_event_id

import mne


def this_function(subject, date, overwrite):
    raw_loaded = False
    for this_contrast in evoked_lcmv_contrasts:
        for evoked_lcmv_weight_norm in evoked_lcmv_weight_norms:
            if evoked_lcmv_weight_norm == 'unit-gain':
                weight_norm = None
            else:
                weight_norm = evoked_lcmv_weight_norm
            for bem_conductivity in bem_conductivities:
                n_layers = len(bem_conductivity)
                output_names = list()
           
                # output_names.append(
                #     fname.source_evoked_beamformer_contrast_simnibs(
                #                     subject=subject,
                #                     date=date,
                #                     fmin=evoked_fmin,
                #                     fmax=evoked_fmax,
                #                     tmin=evoked_tmin,
                #                     tmax=evoked_tmax,
                #                     first_event=this_contrast[0],
                #                     second_event=this_contrast[1],
                #                     reg=evoked_lcmv_regularization,
                #                     weight_norm=evoked_lcmv_weight_norm,
                #                     n_layers=n_layers
                #                         ))
                
                output_names.append( 
                    fname.source_evoked_beamformer_contrast(
                                        subject=subject,
                                        date=date,
                                        fmin=evoked_fmin,
                                        fmax=evoked_fmax,
                                        tmin=evoked_tmin,
                                        tmax=evoked_tmax,
                                        first_event=this_contrast[0],
                                        second_event=this_contrast[1],
                                        reg=evoked_lcmv_regularization,
                                        weight_norm=evoked_lcmv_weight_norm,
                                        n_layers=n_layers
                                            ))
                for output_name in output_names:
                    if should_we_run(output_name, overwrite):
                        if not raw_loaded:
                            raw = \
                                mne.io.read_raw_fif(fname.evoked_filter(
                                    subject=subject,
                                    date=date,
                                    fmin=evoked_fmin,
                                    fmax=evoked_fmax),
                                    preload=False)
                            raw_loaded = True

                        epochs = mne.read_epochs(fname.evoked_epochs_no_proj(
                                        subject=subject,
                                        date=date,
                                        fmin=evoked_fmin,
                                        fmax=evoked_fmax,
                                        tmin=evoked_tmin,
                                        tmax=evoked_tmax),
                            proj=False, preload=False)
            
                        
                        ## apply bads
                        raw.info['bads'] = bad_channels[subject]
                        epochs.info['bads'] = bad_channels[subject]
                        
                        picks = mne.pick_types(epochs.info, 
                                               meg=evoked_lcmv_picks)
                        
                        ## combined values
                        if (this_contrast[0] == 'o0' and \
                            this_contrast[1] == 'o15') or \
                           (this_contrast[0] == 'w0' and \
                            this_contrast[1] == 'w15'):
                           epochs = collapse_event_id(epochs, 
                                                      collapsed_event_id)
                           
                        events = epochs.events
                        baseline = epochs.baseline
                        event_ids = epochs.event_id
                        
                        ## only look at contrast
                        new_event_id = dict()
                        for event in event_ids:
                            if event in this_contrast:
                                new_event_id[event] = event_ids[event]
                                
                        epochs_cov = mne.Epochs(raw, events, new_event_id,
                                                evoked_tmin, evoked_tmax,
                                                baseline,
                                                proj=evoked_lcmv_proj,
                                                preload=True,
                                                picks=picks)
                        
                        ## remove projs
                        epochs.del_proj()
                        epochs_cov.del_proj()
                        rank = None ## for computing covariance
                        
                        ## make forward model on the fly 
                        trans = fname.anatomy_transformation(
                            subject=subject,
                            date=date)
                        src = fname.anatomy_volumetric_source_space(
                                                    subject=subject,
                                                    spacing=src_spacing)
                        if 'simnibs' in output_name:
                            bem = fname.anatomy_simnibs_bem_solutions(
                                subject=subject, n_layers=n_layers)
                        else:
                            bem = fname.anatomy_bem_solutions(
                                                        subject=subject,
                                                        ico=bem_ico,
                                                        n_layers=n_layers)
                        
                        fwd = mne.make_forward_solution(epochs_cov.info,
                                                        trans,
                                                        src, bem)
                            
                        
                        data_cov = mne.compute_covariance(epochs_cov, tmin=0,
                                                          tmax=evoked_tmax,
                                                          rank=rank)
                        filters = mne.beamformer.make_lcmv(epochs_cov.info, fwd,
                                                           data_cov,
                                                           pick_ori='max-power',
                                               weight_norm=weight_norm,
                                               reg=evoked_lcmv_regularization)
                        del epochs_cov ## release memory
                        
                        print('Reconstructing events in contrast: ' + \
                              str(this_contrast))
                            
                        stcs_dict = dict()
                        for event in this_contrast:
                            these_epochs = epochs[event]
                            these_epochs.load_data()
                            these_epochs.pick(picks)
                            evoked = mne.read_evokeds(
                                fname.evoked_average_no_proj(
                                subject=subject, date=date,
                                fmin=evoked_fmin, fmax=evoked_fmax,
                                tmin=evoked_tmin, tmax=evoked_tmax),
                                proj=False,
                                condition=event)
                            # can't add eog and ecg - so remove these
                            bad_channels_evoked = bad_channels[subject].copy()
                            remove_these = ['EOG001', 'EOG002', 'ECG003']
                            for remove_this in remove_these:
                                if remove_this in bad_channels_evoked:
                                    bad_channels_evoked.remove(remove_this)
                            ## ended
                            evoked.info['bads'] = bad_channels_evoked
                            evoked.del_proj()
                            evoked.pick_types(meg=evoked_lcmv_picks)
                            
                            stc = mne.beamformer.apply_lcmv(evoked, filters)
                                
                            if 'simnibs' in output_name:
                                stc.save(
                                    fname.source_evoked_beamformer_simnibs(
                                            subject=subject,
                                            date=date,
                                            fmin=evoked_fmin,
                                            fmax=evoked_fmax,
                                            tmin=evoked_tmin,
                                            tmax=evoked_tmax,
                                            event=event,
                                            reg=evoked_lcmv_regularization,
                                            first_event=this_contrast[0],
                                            second_event=this_contrast[1],
                                            weight_norm=evoked_lcmv_weight_norm,
                                            n_layers=n_layers),
                                    overwrite=True)
                            else:
                                stc.save(fname.source_evoked_beamformer(
                                        subject=subject,
                                        date=date,
                                        fmin=evoked_fmin,
                                        fmax=evoked_fmax,
                                        tmin=evoked_tmin,
                                        tmax=evoked_tmax,
                                        event=event,
                                        reg=evoked_lcmv_regularization,
                                        first_event=this_contrast[0],
                                        second_event=this_contrast[1],
                                        weight_norm=evoked_lcmv_weight_norm,
                                        n_layers=n_layers),
                                overwrite=True)
                            stcs_dict[event] = stc
                            
                        ## calculate ratio between conditions
                        ratio_stc = stcs_dict[this_contrast[0]].copy()
                        ratio_stc._data = (ratio_stc.data - \
                                           stcs_dict[this_contrast[1]].data) / \
                            (ratio_stc.data + stcs_dict[this_contrast[1]].data)
                        if 'simnibs' in output_name:
                            ratio_stc.save(
                            fname.source_evoked_beamformer_contrast_simnibs(
                                       subject=subject,
                                       date=date,
                                       fmin=evoked_fmin,
                                       fmax=evoked_fmax,
                                       tmin=evoked_tmin,
                                       tmax=evoked_tmax,
                                       first_event=this_contrast[0],
                                       second_event=this_contrast[1],
                                       reg=evoked_lcmv_regularization,
                                       weight_norm=evoked_lcmv_weight_norm,
                                       n_layers=n_layers),
                                overwrite=True)
                        else:
                            ratio_stc.save(
                                fname.source_evoked_beamformer_contrast(
                                   subject=subject,
                                   date=date,
                                   fmin=evoked_fmin,
                                   fmax=evoked_fmax,
                                   tmin=evoked_tmin,
                                   tmax=evoked_tmax,
                                   first_event=this_contrast[0],
                                   second_event=this_contrast[1],
                                   reg=evoked_lcmv_regularization,
                                   weight_norm=evoked_lcmv_weight_norm,
                                   n_layers=n_layers),
                            overwrite=True)
                    
if submitting_method == 'hyades_frontend':
    queue = 'highmem_short.q'
    job_name = 'elcmv'
    n_jobs = 4
    deps = ['eve', 'efilt', 'eepo', 'eave', 'mri', 'ana', 'fwd', 'snibs', 
            'snbem']

if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))                                 