#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:54:18 2021

@author: lau

Config file for functional significance of the cerebellar clock

"""

#%% IMPORTS

from os import getlogin, environ
from socket import getfqdn
from fnames import FileNames

#%% GET USER AND HOST AND SET PROJECT PATH
try:
    user = getlogin()
except OSError: # on hyades
    user = None
host = getfqdn()


if user == 'lau' and host == 'lau':
    ## my laptop
    project_path = '/home/lau/projects/functional_cerebellum'
    submitting_method = 'local'
elif user is None and host[:6] == 'hyades':
    hyades_core = int(host[6:8])
    project_path = '/projects/MINDLAB2021_MEG-CerebellarClock-FuncSig'
    if hyades_core < 4:
        ## CFIN server frontend
        submitting_method = 'hyades_frontend'
    else:
        ## CFIN server backend
        submitting_method = 'hyades_backend'
else:
    raise RuntimeError('Please edit config.py to include this "user" and '
                       '"host"')
    
#%% SUBJECT SPECIFIC

bad_channels = dict()
behavioural_data_time_stamps = dict()

bad_channels['0002'] = ['MEG1531']
bad_channels['0003'] = ['MEG2631']

behavioural_data_time_stamps['0002'] = '104645'
behavioural_data_time_stamps['0003'] = '135455'

#%% GENERAL

collapsed_event_id = dict(w0=dict(old_event_ids=['w0_hit', 'w0_miss'],
                                 new_event_id=dict(w0=81)),
                       w15=dict(old_event_ids=['w15_hit', 'w15_miss'],
                                  new_event_id=dict(w15=97)),
                       o0=dict(old_event_ids=['o0_cr', 'o0_fa'],
                                   new_event_id=dict(o0=144)),
                       o15=dict(old_event_ids=['o15_cr', 'o15_fa'],
                                    new_event_id=dict(o15=160)))

#%% EVOKED ANALYSIS

## filtering

evoked_fmin = None
evoked_fmax = 40 # Hz

## epoching

evoked_tmin = -0.200 # s
evoked_tmax =  1.000 # s
evoked_baseline = (None, 0) # s
evoked_decim = 4
evoked_event_id = dict(s1=3, s2=5, s3=9,
                       s4_0=19, s5_0=21, s6_0=25,
                       s4_15=35, s5_15=37, s6_15=41,
                       w0_hit=337, w15_hit=353,
                       o0_cr=400, o15_cr=416,
                       w0_miss=593, w15_miss=609,
                       o0_fa=656, o15_fa=672)
evoked_reject = dict(mag=4e-12, grad=4000e-13) # T / T/cm
evoked_proj = False

## averaging

#%% HILBERT ANALYSIS

## filtering

hilbert_fmins = [4, 14] # Hz
hilbert_fmaxs = [7, 30] # Hz

## transforming

hilbert_tmin = -0.750 # s
hilbert_tmax =  0.750 # s
hilbert_baseline = None
hilbert_decim = 1
hilbert_proj = False
hilbert_event_id = evoked_event_id
hilbert_reject = None ## think about this...

## averaging

## z transform contrasts

hilbert_contrasts = [
                     ['s1', 's2'], ['s2', 's3'], ['s3', 's4_0'],
                     ['s4_0', 's5_0'], ['s5_0', 's6_0'],
                     ['s4_15', 's5_15'], ['s5_15', 's6_15'],
                     ['s4_0', 's4_15'], ['s5_0', 's5_15'], ['s6_0', 's6_15'],
                     ['w0', 'w15'],
                     ['w0_miss', 'w15_miss'], ['w0_hit', 'w15_hit'],
                     ['w0_hit', 'w0_miss'], ['w15_hit', 'w15_miss'],
                     ['o0', 'o15'],
                     ['o0_fa', 'o15_fa'], ['o0_cr', 'o15_cr'],
                     ['o0_cr', 'o0_fa'], ['o15_cr', 'o15_fa']
                     
                    ]

#%% CREATE FORWARD MODEL

## import mri

t1_file_ending = 't1_mprage_3d_sag_fatsat'
t2_file_ending = 't2_tse_sag_HighBW'

## freesurfer reconstruction

n_jobs_freesurfer = 3

## watershed - is there a better alternative with T1 and T2 combined?

## make scalp surface with fine resolution



## transformation

## volumetric source space

src_spacing = 7.5 # mm

## bem model

bem_ico = 4
bem_conductivity = [0.3]# , 0.006, 0.3] # should we do three-layer instead?

## bem solution

## morph 

morph_subject_to = 'fsaverage'

## forward solution

#%% SOURCE ANALYSIS EVOKED

## lcmv contrasts

evoked_lcmv_contrasts = hilbert_contrasts
evoked_lcmv_weight_norm = 'unit-noise-gain'
evoked_lcmv_regularization = 0.00 # should we regularize?
evoked_lcmv_picks = 'mag' # can they be combined?

## morph contrasts

#%% SOURCE ANALYSIS HILBERT

## lcmv contrasts

hilbert_lcmv_contrasts = hilbert_contrasts
hilbert_lcmv_weight_norm = 'unit-noise-gain'
hilbert_lcmv_regularization = 0.05 # should we regularize?
hilbert_lcmv_picks = 'mag' # can they be combined?

## morph contrasts

#%% GRAND AVERAGE AND STAISTICS EVOKED

## grand average

## statistics

#%% GRAND AVERAGE AND STAISTICS HILBERT

## grand average

## statistics

#%% SET FILENAMES

fname = FileNames()    

## directories
fname.add('project_path', project_path)
fname.add('raw_path', '{project_path}/raw')
fname.add('scratch_path', '{project_path}/scratch')
fname.add('MEG_path', '{scratch_path}/MEG')
fname.add('subjects_dir', '{scratch_path}/freesurfer')
fname.add('figures_path', '{scratch_path}/figures')
fname.add('behavioural_path', '{scratch_path}/behavioural_data')
fname.add('script_path', '{project_path}/scripts')
fname.add('python_path', '{script_path}/python')
fname.add('python_qsub_path', '{python_path}/qsub')

## directories that require input
fname.add('subject_path', '{MEG_path}/{subject}/{date}')
fname.add('subject_figure_path', '{figures_path}/{subject}/{date}')
fname.add('subject_behaviour_path', '{behavioural_path}/{subject}/{date}')
fname.add('subject_freesurfer_path', '{subjects_dir}/{subject}') 
fname.add('subject_bem_path', '{subject_freesurfer_path}/bem')
fname.add('subject_beamformer_evoked_path',
          '{subject_path}/beamformer_evoked')
fname.add('subject_beamformer_hilbert_path',
          '{subject_path}/beamformer_hilbert')
fname.add('subject_MR_path', '{raw_path}/{subject}/{date}/MR')

## raw filenames
fname.add('trigger_test', '{raw_path}/{subject}/{date}/MEG/001.trigger_test/'
                          'files/trigger_test.fif')
fname.add('raw_file', '{raw_path}/{subject}/{date}/MEG/'
                      '001.func_cerebellum_raw/files/func_cerebellum_raw.fif')
fname.add('behavioural_data', '{subject_behaviour_path}/{subject_code}_'
                              '{date_short}_{time_stamp}_data.csv')

## MEG output
fname.add('events', '{subject_path}/fc-eve.fif')

## evoked
fname.add('evoked_filter', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz-raw.fif')
fname.add('evoked_epochs_no_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-no_proj-epo.fif')
fname.add('evoked_epochs_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-proj-epo.fif')
fname.add('evoked_average_no_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-no_proj-ave.fif')
fname.add('evoked_average_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-proj-ave.fif')

## hilbert
fname.add('hilbert_filter', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz-raw.fif')
fname.add('hilbert_epochs_no_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-no_proj-epo.fif')
fname.add('hilbert_epochs_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-proj-epo.fif')

fname.add('hilbert_average_no_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-no_proj-ave.fif')
fname.add('hilbert_average_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                   '-{tmin}-{tmax}-s-proj-ave.fif')

fname.add('hilbert_wilcoxon_no_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                '-{tmin}-{tmax}-s-no_proj-z_contrast-ave.fif')
fname.add('hilbert_wilcoxon_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                '-{tmin}-{tmax}-s-proj-z_contrast-ave.fif')

## anatomy
fname.add('anatomy_bem_surfaces', '{subject_bem_path}/ico-{ico}-bem.fif')
fname.add('anatomy_bem_solutions', '{subject_bem_path}/ico-{ico}-bem-sol.fif')
fname.add('anatomy_volumetric_source_space', '{subject_bem_path}/volume'
                                            '-{spacing}_mm-src.fif')
fname.add('anatomy_morph_volume', '{subject_bem_path}/volume-{spacing}_mm'
                                    '-morph.h5')
fname.add('anatomy_transformation', '{subject_bem_path}/fc-trans.fif')
fname.add('anatomy_forward_model', '{subject_path}/fc-volume-{spacing}_mm-fwd.fif')

## source evoked
fname.add('source_evoked_beamformer', '{subject_path}/beamformer_evoked/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-vl.stc')
fname.add('source_evoked_beamformer_contrast', '{subject_path}'
          '/beamformer_evoked/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-'
          'reg-{reg}-contrast-{first_event}-versus-{second_event}-vl.stc')

## source hilbert
fname.add('source_hilbert_beamformer', '{subject_path}/beamformer_hilbert/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-vl.stc')
fname.add('source_hilbert_beamformer_contrast', '{subject_path}'
          '/beamformer_hilbert/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-'
          'reg-{reg}-contrast-{first_event}-versus-{second_event}-vl.stc')



## figure names
fname.add('events_plot', '{subject_figure_path}/fc_events.png')

## set the environment for FreeSurfer and MNE-Python
environ["SUBJECTS_DIR"] = fname.subjects_dir
