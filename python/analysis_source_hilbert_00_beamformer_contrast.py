#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 17 15:02:13 2021

@author: lau
"""

from config import (fname, submitting_method, hilbert_lcmv_contrasts,
                    hilbert_lcmv_weight_norm, hilbert_lcmv_regularization,
                    hilbert_lcmv_picks, hilbert_fmins, hilbert_fmaxs,
                    hilbert_tmin, hilbert_tmax, hilbert_proj, bad_channels,
                    collapsed_event_id, src_spacing, bem_ico)
from sys import argv
from helper_functions import should_we_run, collapse_event_id

import mne
import numpy as np

# hilbert_lcmv_contrasts = hilbert_lcmv_contrasts[-1:]

def beamformer_contrast(subject, date, overwrite):
    
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        raw_loaded = False
        for this_contrast in hilbert_lcmv_contrasts:
            output_name = \
                fname.source_hilbert_beamformer_contrast(subject=subject,
                                                         date=date,
                                                         fmin=fmin,
                                                         fmax=fmax,
                                                         tmin=hilbert_tmin,
                                                         tmax=hilbert_tmax,
                                            first_event=this_contrast[0],
                                            second_event=this_contrast[1],
                                            reg=hilbert_lcmv_regularization)
                
            if should_we_run(output_name, overwrite):
                if not raw_loaded:
                    raw = \
                    mne.io.read_raw_fif(fname.hilbert_filter(subject=subject,
                                                                 date=date,
                                                                 fmin=fmin,
                                                                 fmax=fmax),
                                            preload=False)
                epochs_hilbert = \
                mne.read_epochs(fname.hilbert_epochs_no_proj(subject=subject,
                                                                date=date,
                                                                fmin=fmin,
                                                                fmax=fmax,
                                                            tmin=hilbert_tmin,
                                                            tmax=hilbert_tmax),
                                    proj=False, preload=False)
                ## apply bads
                raw.info['bads'] = bad_channels[subject]
                epochs_hilbert.info['bads'] = bad_channels[subject]
                
                picks = mne.pick_types(epochs_hilbert.info,
                                       meg=hilbert_lcmv_picks)
                
                
                ## combine values
                if (this_contrast[0] == 'o0' and \
                    this_contrast[1] == 'o15') or \
                   (this_contrast[0] == 'w0' and this_contrast[1] == 'w15'):
                    epochs_hilbert = collapse_event_id(epochs_hilbert,
                                                       collapsed_event_id)
                events = epochs_hilbert.events
                baseline = epochs_hilbert.baseline
                event_ids = epochs_hilbert.event_id
                
                ## only look at contrast
                new_event_id = dict()
                for event in event_ids:
                    if event in this_contrast:
                        new_event_id[event] = event_ids[event]
                        
                epochs_cov = mne.Epochs(raw, events, new_event_id,
                                        hilbert_tmin, hilbert_tmax, baseline,
                                        proj=hilbert_proj, preload=True,
                                        picks=picks)
                
                ## remove projs
                epochs_hilbert.del_proj()
                epochs_cov.del_proj()
                rank = None ## for computing covariance
                
                ## make forward model on the fly (only first time)
                if not raw_loaded:
                    trans = fname.anatomy_transformation(subject=subject)
                    src = fname.anatomy_volumetric_source_space(
                                                        subject=subject,
                                                        spacing=src_spacing)
                    bem = fname.anatomy_bem_solutions(subject=subject,
                                                      ico=bem_ico)
                    
                    fwd = mne.make_forward_solution(epochs_cov.info, trans,
                                                    src, bem)
                    
                    raw_loaded = True
                
                if baseline is None:
                    data_cov = mne.compute_covariance(epochs_cov, tmin=None,
                                                      tmax=None, rank=rank)
                else:
                    raise RuntimeError('"baseline" ' + str(baseline) + 
                                       ' not implemented')    
                    
                filters = mne.beamformer.make_lcmv(epochs_cov.info, fwd,
                                                   data_cov,
                                                   pick_ori='max-power',
                                       weight_norm=hilbert_lcmv_weight_norm,
                                       reg=hilbert_lcmv_regularization)
                del epochs_cov ## release memory
                
                print('Reconstructing events in contrast: ' + \
                      str(this_contrast))
                
                stcs_dict = dict()
                
                for event in this_contrast:
                    these_epochs = epochs_hilbert[event]
                    these_epochs.load_data()
                    these_epochs.pick(picks)
                    
                    stcs = mne.beamformer.apply_lcmv_epochs(these_epochs,
                                                            filters)
                    for stc in stcs:
                        stc._data = np.array(np.abs(stc.data),
                                             dtype='float64')
                    
                    stc_mean = stcs[0].copy()
                    mean_data = np.mean([stc.data for stc in stcs], axis=0)
                    stc_mean._data = mean_data
                    stc_mean.save(fname.source_hilbert_beamformer(
                                                          subject=subject,
                                                          date=date,
                                                          fmin=fmin,
                                                          fmax=fmax,
                                                          tmin=hilbert_tmin,
                                                          tmax=hilbert_tmax,
                                                          event=event,
                                          reg=hilbert_lcmv_regularization,
                                          first_event=this_contrast[0],
                                          second_event=this_contrast[1]))
                    stcs_dict[event] = stc_mean
                    
                ## calculate ratio between conditions
                
                ratio_stc = stcs_dict[this_contrast[0]].copy()
                ratio_stc._data = (ratio_stc.data - \
                                   stcs_dict[this_contrast[1]].data) / \
                    (ratio_stc.data + stcs_dict[this_contrast[1]].data)
                    
                ratio_stc.save(fname.source_hilbert_beamformer_contrast(
                           subject=subject,
                           date=date,
                           fmin=fmin,
                           fmax=fmax,
                           tmin=hilbert_tmin,
                           tmax=hilbert_tmax,
                           first_event=this_contrast[0],
                           second_event=this_contrast[1],
                           reg=hilbert_lcmv_regularization))
                
        ## memory control
        del stcs, stcs_dict, ratio_stc
        
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'hlcmv'
    n_jobs = 3

if submitting_method == 'hyades_backend':
    print(argv[:])
    beamformer_contrast(subject=argv[1], date=argv[2],
                        overwrite=bool(int(argv[3])))              