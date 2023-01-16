#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 14:54:18 2021

@author: lau

Config file for functional significance of the cerebellar clock

"""

#%% IMPORTS

from os import getlogin, environ
from os.path import join
from socket import getfqdn
from fnames import FileNames
import numpy as np

import warnings ## to ignore future warning from nilearn
warnings.simplefilter(action='ignore', category=FutureWarning)

from nilearn import datasets

#%% GET USER AND HOST AND SET PROJECT PATH
try:
    user = getlogin()
except OSError: # on hyades
    user = None
host = getfqdn()

project_name = 'MINDLAB2021_MEG-CerebellarClock-FuncSig'


if user == 'lau' and host == 'lau':
    ## my laptop
    project_path = '/home/lau/projects/functional_cerebellum'
    submitting_method = 'local'
elif (user is None or user == 'lau') and host[:6] == 'hyades':
    hyades_core = int(host[6:8])
    project_path = join('/projects/', project_name)
    if hyades_core < 4:
        ## CFIN server frontend
        submitting_method = 'hyades_frontend'
    else:
        ## CFIN server backend
        submitting_method = 'hyades_backend'
else:
    raise RuntimeError('Please edit config.py to include this "user" and '
                       '"host"')

#%% RECORDINGS

recordings = [
    dict(subject='0001', date='20210810_000000', mr_date='20191015_121553',
         storm_code='MZH'),
    dict(subject='0002', date='20210804_000000', mr_date='20191015_112257',
         storm_code='DU9'),
    dict(subject='0003', date='20210802_000000', mr_date='20210812_102146',
         storm_code='WE0'),
    dict(subject='0004', date='20210728_000000', mr_date='20210811_164949',
         storm_code='VVY'),
    dict(subject='0005', date='20210728_000000', mr_date='20210816_091907',
         storm_code='MM4'),
    dict(subject='0006', date='20210728_000000', mr_date='20210811_173642',
         storm_code='GVY'), 
    dict(subject='0007', date='20210728_000000', mr_date='20210812_105728',
         storm_code='SAM'),
    dict(subject='0008', date='20210730_000000', mr_date='20210812_081520',
         storm_code='SWR'),
    dict(subject='0009', date='20210730_000000', mr_date='20210812_141341',
         storm_code='POM'),
    dict(subject='0010', date='20210730_000000', mr_date='20210812_094201',
         storm_code='XR2'),
    dict(subject='0011', date='20210730_000000', mr_date='20191015_104445',
         storm_code='PPH'),
    dict(subject='0012', date='20210802_000000', mr_date='20210812_145235',
         storm_code='CBU'),
    dict(subject='0013', date='20210802_000000', mr_date='20210811_084903',
         storm_code='MUN'),
    dict(subject='0014', date='20210802_000000', mr_date='20210812_164859',
         storm_code='KLZ'),
    dict(subject='0015', date='20210804_000000', mr_date='20210811_133830',
         storm_code='431'),
    dict(subject='0016', date='20210804_000000', mr_date='20210812_153043',
         storm_code='HZ0'),
    dict(subject='0017', date='20210805_000000', mr_date='20210820_123549',
         storm_code='UHJ'),
    dict(subject='0018', date='20210805_000000', mr_date='20210811_113632',
         storm_code='JMU'),
    dict(subject='0019', date='20210805_000000', mr_date='20210811_101021',
         storm_code='A39'),
    dict(subject='0020', date='20210806_000000', mr_date='20210812_085148',
         storm_code='EC0'),
    dict(subject='0021', date='20210806_000000', mr_date='20210811_145727',
         storm_code='JJK'),
    dict(subject='0022', date='20210806_000000', mr_date='20210811_141117',
         storm_code='OWN'),
    dict(subject='0023', date='20210809_000000', mr_date='20210812_112225',
         storm_code='IAG'),
    dict(subject='0024', date='20210809_000000', mr_date='20210812_125146',
         storm_code='ORR'),
    dict(subject='0026', date='20210810_000000', mr_date='20210811_120947',
         storm_code='TDY'),
    dict(subject='0027', date='20210810_000000', mr_date='20210811_105000',
         storm_code='OZH'),
    dict(subject='0028', date='20210817_000000', mr_date='20210820_111354',
         storm_code='EUG'),
    dict(subject='0029', date='20210817_000000', mr_date='20210820_103315',
         storm_code='J5A'),
    dict(subject='0030', date='20210817_000000', mr_date='20210820_085929',
         storm_code='OSD'),
    dict(subject='0031', date='20210825_000000', mr_date='20210820_094714',
         storm_code='M5F'),
             ]    
    
#%% SUBJECT SPECIFIC

bad_channels = dict()
behavioural_data_time_stamps = dict()

bad_channels['0001'] = ['MEG0232', 'MEG0321', 'MEG0422', 'MEG2613']
bad_channels['0002'] = ['MEG0121', 'MEG0422', 'MEG0441', 'MEG1133', 'MEG2613']
bad_channels['0003'] = ['MEG0321', 'MEG0422', 'MEG1133', 'MEG2523']
bad_channels['0004'] = ['MEG0411', 'MEG0422', 'MEG1133', 'MEG2613']
bad_channels['0005'] = ['MEG0422', 'MEG2521', 'MEG2542', 'MEG2613']
bad_channels['0006'] = ['MEG0422']
bad_channels['0007'] = ['MEG0422', 'MEG2613']
bad_channels['0008'] = ['MEG0422', 'MEG1343', 'MEG2613']
bad_channels['0009'] = ['MEG0422', 'MEG2413', 'MEG2613']
bad_channels['0010'] = ['EOG001', 'EOG002', 'MEG0422']
bad_channels['0011'] = ['MEG0422']
bad_channels['0012'] = ['MEG0221','MEG0422', 'MEG2613']
bad_channels['0013'] = ['MEG0422', 'MEG0932', 'MEG2613']
bad_channels['0014'] = ['MEG0321', 'MEG0422', 'MEG0811', 'MEG2613']
bad_channels['0015'] = ['MEG0422', 'MEG1133', 'MEG2613']
bad_channels['0016'] = ['MEG0422', 'MEG1133', 'MEG2613']
bad_channels['0017'] = ['MEG0422', 'MEG0613', 'MEG1133', 'MEG2613']
bad_channels['0018'] = ['ECG003', 'MEG0422', 'MEG0613', 'MEG1133', 'MEG2613']
bad_channels['0019'] = ['MEG0422', 'MEG0613', 'MEG1133', 'MEG2613']
bad_channels['0020'] = ['MEG0422', 'MEG0811', 'MEG2613']
bad_channels['0021'] = ['MEG0321', 'MEG0422', 'MEG0811', 'MEG2613']
bad_channels['0022'] = ['MEG0422', 'MEG2613']
bad_channels['0023'] = ['MEG0422', 'MEG1613']
bad_channels['0024'] = ['MEG0422', 'MEG2613']
bad_channels['0026'] = ['MEG0422', 'MEG2613']
bad_channels['0027'] = ['MEG0232', 'MEG0422', 'MEG2613']
bad_channels['0028'] = ['ECG003', 'MEG0422', 'MEG2613']
bad_channels['0029'] = ['MEG0422', 'MEG0921', 'MEG1431', 'MEG2613']
bad_channels['0030'] = ['MEG0422', 'MEG1643', 'MEG2613']
bad_channels['0031'] = ['MEG0422', 'MEG1423', 'MEG2613']

behavioural_data_time_stamps['0001'] = '075323'
behavioural_data_time_stamps['0002'] = '075218'
behavioural_data_time_stamps['0003'] = '133323'
behavioural_data_time_stamps['0004'] = '074414'
behavioural_data_time_stamps['0005'] = '110852'
behavioural_data_time_stamps['0006'] = '134437'
behavioural_data_time_stamps['0007'] = '153731'
behavioural_data_time_stamps['0008'] = '084401'
behavioural_data_time_stamps['0009'] = '111206'
behavioural_data_time_stamps['0010'] = '131907'
behavioural_data_time_stamps['0011'] = '163628'
behavioural_data_time_stamps['0012'] = '085256'
behavioural_data_time_stamps['0013'] = '100241'
behavioural_data_time_stamps['0014'] = '160424'
behavioural_data_time_stamps['0015'] = '124445'
behavioural_data_time_stamps['0016'] = '153734'
behavioural_data_time_stamps['0017'] = '075941'
behavioural_data_time_stamps['0018'] = '103918'
behavioural_data_time_stamps['0019'] = '122553'
behavioural_data_time_stamps['0020'] = '095715'
behavioural_data_time_stamps['0021'] = '135334'
behavioural_data_time_stamps['0022'] = '155547'
behavioural_data_time_stamps['0023'] = '081423'
behavioural_data_time_stamps['0024'] = '103941'
behavioural_data_time_stamps['0026'] = '101724'
behavioural_data_time_stamps['0027'] = '122355'
behavioural_data_time_stamps['0028'] = '102259'
behavioural_data_time_stamps['0029'] = '131300'
behavioural_data_time_stamps['0030'] = '161750'
behavioural_data_time_stamps['0031'] = '085032'

#%% GENERAL

collapsed_event_id = dict(w0=dict(old_event_ids=['w0_hit', 'w0_miss'],
                                 new_event_id=dict(w0=81)),
                       w15=dict(old_event_ids=['w15_hit', 'w15_miss'],
                                  new_event_id=dict(w15=97)),
                       o0=dict(old_event_ids=['o0_cr', 'o0_fa'],
                                   new_event_id=dict(o0=144)),
                       o15=dict(old_event_ids=['o15_cr', 'o15_fa'],
                                    new_event_id=dict(o15=160)))

split_recording_subjects = ['0006']
subjects_with_MRs_from_elsewhere = ['0001', '0002', '0011']
subjects_with_no_T2 = ['0007']
subjects_with_no_3_layer_BEM_watershed = ['0001', '0004', '0008'
                                          '0011', '0012', '0022',
                                          '0028', '0029']
subjects_with_no_BEM_simnibs = ['0010', '0029']
subjects_missing_n_trials = dict()
subjects_missing_n_trials['0008'] = 111 # only the last 1989 events, recorded

bad_subjects = ['0006', '0011']

#%% GENERAL PLOTTING

n_jobs_power_spectra = 3

#%% EVOKED ANALYSIS

## filtering

evoked_fmin = None
evoked_fmax = 40 # Hz

## epoching

evoked_tmin = -0.200 # s
evoked_tmax =  1.000 # s
evoked_baseline = (None, 0) # s
evoked_decim = 1
evoked_event_id = dict(s1=3, s2=5, s3=9,
                       s4_0=19, s5_0=21, s6_0=25,
                       s4_15=35, s5_15=37, s6_15=41,
                       w0_hit=337, w15_hit=353,
                       o0_cr=400, o15_cr=416,
                       w0_miss=593, w15_miss=609,
                       o0_fa=656, o15_fa=672)
evoked_reject = dict(mag=4e-12, grad=4000e-13) # T / T/cm
# evoked_proj = False

## averaging


#%% TFR ANALYSIS

## epoching

tfr_tmin = -0.750 # s
tfr_tmax =  0.750 # s
tfr_baseline = (None, None)
tfr_decim = 4
tfr_event_id = dict(s1=3, s2=5, s3=9,
                       s4_0=19, s5_0=21, s6_0=25,
                       s4_15=35, s5_15=37, s6_15=41,
                       w0_hit=337, w15_hit=353,
                       o0_cr=400, o15_cr=416,
                       w0_miss=593, w15_miss=609,
                       o0_fa=656, o15_fa=672)
tfr_reject = dict(mag=4e-12, grad=4000e-13) # T / T/cm

## average

tfr_freqs = np.arange(2, 41)
tfr_n_cycles = tfr_freqs
tfr_n_jobs = 2


#%% HILBERT ANALYSIS

## filtering

hilbert_fmins = [4, 8,  14]#, 15] # Hz - 14-12 added to emulate the faulty
hilbert_fmaxs = [7, 12, 30]#, 11] # Hz - analysis from the cerebellar clock paper

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
                      ['s1', 's2'], ['s2', 's3'], #['s3', 's4_0'],
                      # ['s4_0', 's5_0'], ['s5_0', 's6_0'],
                      # ['s4_15', 's5_15'], ['s5_15', 's6_15'],
                      # ['s4_0', 's4_15'], ['s5_0', 's5_15'], ['s6_0', 's6_15'],
                     ['w0', 'w15'],
                       # ['w0_miss', 'w15_miss'], ['w0_hit', 'w15_hit'],
                       # ['w0_hit', 'w0_miss'], ['w15_hit', 'w15_miss'],
                     ['o0', 'o15'],
                      #  ['o0_fa', 'o15_fa'], ['o0_cr', 'o15_cr'],
                      # ['o0_cr', 'o0_fa'], ['o15_cr', 'o15_fa']
                     
                    ]

#%% CREATE FORWARD MODEL

## import mri

t1_file_ending = 't1_mprage_3D_sag_fatsat'
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
bem_conductivities = [
                        [0.3], # single-layer model
                        [0.3, 0.006, 0.3] # three-layer model
                        ]

## bem solution

## morph 

morph_subject_to = 'fsaverage'

## forward solution

n_jobs_forward = 2

## simnibs

stls = ['csf.stl', 'skull.stl', 'skin.stl'] ## from simnibs/mri2mesh

#%% SOURCE ANALYSIS EVOKED

## lcmv contrasts

evoked_lcmv_contrasts = hilbert_contrasts
evoked_lcmv_weight_norms = ['unit-noise-gain-invariant', 'unit-gain']
evoked_lcmv_regularization = 0.00 # should we regularize?
evoked_lcmv_picks = 'mag' # can they be combined?
evoked_lcmv_proj = False

## morph contrasts

#%% SOURCE ANALYSIS HILBERT

## lcmv contrasts

hilbert_lcmv_contrasts = hilbert_contrasts
hilbert_lcmv_weight_norms = ['unit-noise-gain-invariant', 'unit-gain']
hilbert_lcmv_regularization = 0.00 # should we regularize?
hilbert_lcmv_picks = 'mag' # can they be combined?

## morph contrasts

## labels

hilbert_atlas_contrasts = [
                      ['s1', 's2'], ['s2', 's3'], 
                      #['s3', 's4_0'],
                      # ['s4_0', 's5_0'], ['s5_0', 's6_0'],
                      # ['s4_15', 's5_15'], ['s5_15', 's6_15'],
                      # ['s4_0', 's4_15'], ['s5_0', 's5_15'], ['s6_0', 's6_15'],
                     ['w0', 'w15'],
                       # ['w0_miss', 'w15_miss'], ['w0_hit', 'w15_hit'],
                       # ['w0_hit', 'w0_miss'], ['w15_hit', 'w15_miss'],
                     ['o0', 'o15'],
                      #  ['o0_fa', 'o15_fa'], ['o0_cr', 'o15_cr'],
                      # ['o0_cr', 'o0_fa'], ['o15_cr', 'o15_fa']
                     
                    ]
atlas = datasets.fetch_atlas_aal()
rois = [
        'Postcentral_L',
        'Postcentral_R',
        'Parietal_Inf_L',
        'Parietal_Inf_R',
         'Thalamus_L',
         'Thalamus_R',
         'Putamen_L',
         'Putamen_R',
         'Cerebelum_Crus1_L',
         'Cerebelum_Crus1_R',
         'Cerebelum_Crus2_L',
         'Cerebelum_Crus2_R',
         'Cerebelum_3_L',
         'Cerebelum_3_R',
         'Cerebelum_4_5_L',
         'Cerebelum_4_5_R',
         'Cerebelum_6_L',
         'Cerebelum_6_R',
         'Cerebelum_7b_L',
         'Cerebelum_7b_R',
         'Cerebelum_8_L',
         'Cerebelum_8_R',
         'Cerebelum_9_L',
         'Cerebelum_9_R',
         'Cerebelum_10_L',
         'Cerebelum_10_R',
         'Vermis_1_2',
         'Vermis_3',
         'Vermis_4_5',
         'Vermis_6',
         'Vermis_7',
         'Vermis_8',
         'Vermis_9',
         'Vermis_10'
]

#%% ENVELOPE CORRELATIONS

envelope_events = [['w0', 'w15'],
                   ['o0', 'o15']]
envelope_downsampling = 100 ## ?!
envelope_fmins = [4, 14]
envelope_fmaxs = [7, 30]
envelope_tmin = -0.100
envelope_tmax =  0.100
envelope_weight_norm = 'unit-noise-gain-invariant'
envelope_regularization = 0.00
envelope_picks = 'mag' # can they be combined?

subjects_conn_cannot_be_saved = ['0005', '0008', 
                                 '0015', '0016', '0017', '0018']


#%% GRAND AVERAGE AND STATISTICS EVOKED

## grand average

## statistics

evoked_lcmv_stat_contrasts = [ 
                ['s1', 's2'],  ['s2',  's3'],
                ['w0', 'w15'], ['o0', 'o15']
                ]
evoked_lcmv_stat_tmin = 0 # s
evoked_lcmv_stat_tmax = 0.400 # what should it be? 0.600 # s
evoked_lcmv_stat_p = 0.05
evoked_lcmv_stat_n_permutations = 1024
evoked_lcmv_stat_n_jobs = 12
evoked_lcmv_stat_seed = 7
evoked_lcmv_stat_connectivity_dist = None # 

#%% GRAND AVERAGE AND STATISTICS HILBERT

## grand average

## statistics - sensor space

hilbert_stat_contrasts = [ 
                ['s1', 's2'],  ['s2',  's3'],
                ['w0', 'w15'], ['o0', 'o15']
                ]
hilbert_stat_tmin = -0.400 # s
hilbert_stat_tmax = 0.400 # s
hilbert_stat_p = 0.05
hilbert_stat_n_permutations = 1024
hilbert_stat_n_jobs = 4
hilbert_stat_seed = 7
hilbert_stat_connectivity_dist = None # use direct neighbours #0.075 # m
hilbert_stat_channels = 'mag'

## statistics - source space

hilbert_lcmv_stat_contrasts = [ 
                ['s1', 's2'],  ['s2',  's3'],
                ['w0', 'w15'], ['o0', 'o15']
                ]
hilbert_lcmv_stat_tmin = 0.000 # s
hilbert_lcmv_stat_tmax = 0.200 # s
hilbert_lcmv_stat_p = 0.05
hilbert_lcmv_stat_n_permutations = 1024
hilbert_lcmv_stat_n_jobs = 12
hilbert_lcmv_stat_seed = 7
hilbert_lcmv_stat_connectivity_dist = None # 

#%% SET FILENAMES

fname = FileNames()    

## directories
fname.add('project_path', project_path)
fname.add('raw_path', '{project_path}/raw')
fname.add('scratch_path', '{project_path}/scratch')
fname.add('MEG_path', '{scratch_path}/MEG')
fname.add('subjects_dir', '{scratch_path}/freesurfer')
fname.add('simnibs_subjects_dir', '{scratch_path}/simnibs')
fname.add('figures_path', '{scratch_path}/figures')
fname.add('behavioural_path', '{scratch_path}/behavioural_data')
fname.add('script_path', '{project_path}/scripts')
fname.add('python_path', '{script_path}/python')
fname.add('python_qsub_path', '{python_path}/qsub')

## SimNIBS directories

fname.add('subject_simnibs_path', '{simnibs_subjects_dir}/{subject}')
fname.add('simnibs_freesurfer_subjects_dir',
          '{simnibs_subjects_dir}/freesurfer')
fname.add('subject_fs_path', '{subject_simnibs_path}/fs_{subject}')
fname.add('subject_m2m_path', '{subject_simnibs_path}/m2m_{subject}')
fname.add('simnibs_bem_path', '{subject_fs_path}/bem')

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
fname.add('subject_beamformer_hilbert_labels_path',
          '{subject_path}/beamformer_hilbert/labels')
fname.add('subject_envelope_path', '{subject_path}/envelopes')
fname.add('subject_MR_path', '{raw_path}/{subject}/{date}/MR')
fname.add('subject_MR_elsewhere_path',
          '{scratch_path}/MRs_from_elsewhere/{subject}/{date}/MR')

## raw filenames
fname.add('trigger_test', '{raw_path}/{subject}/{date}/MEG/001.trigger_test/'
                          'files/trigger_test.fif')
fname.add('raw_file', '{raw_path}/{subject}/{date}/MEG/'
                      '001.func_cerebellum_raw/files/func_cerebellum_raw.fif')
fname.add('split_raw_file_1', '{raw_path}/{subject}/{date}/MEG/'
         '001.func_cerebellum_raw_1/files/func_cerebellum_raw_1.fif')
fname.add('split_raw_file_2', '{raw_path}/{subject}/{date}/MEG/'
         '002.func_cerebellum_raw_2/files/func_cerebellum_raw_2.fif')
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
fname.add('evoked_grand_average_proj_interpolated',
              '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
              '-{tmin}-{tmax}-s-proj-interpolated-ave.fif')
fname.add('evoked_grand_average_proj', '{subject_path}/fc-filt'
                              '-{fmin}-{fmax}-Hz'
                              '-{tmin}-{tmax}-s-proj-ave.fif')


## tfr
fname.add('tfr_epochs', '{subject_path}/fc-no-filt-{tmin}-{tmax}-s'
                          '-proj-epo.fif')
fname.add('tfr_average', '{subject_path}/fc-no-filt'
                                   '-{tmin}-{tmax}-s-tfr.h5')
fname.add('tfr_grand_average_interpolated',
              '{subject_path}/fc-no-filt'
              '-{tmin}-{tmax}-s-interpolated-tfr.h5')
fname.add('tfr_grand_average', '{subject_path}/fc-no-filt'
                              '-{tmin}-{tmax}-s-tfr.h5')

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
                                '-{tmin}-{tmax}-s-no_proj-z_contrasts-ave.fif')
fname.add('hilbert_wilcoxon_proj', '{subject_path}/fc-filt-{fmin}-{fmax}-Hz'
                                '-{tmin}-{tmax}-s-proj-z_contrasts-ave.fif')

fname.add('hilbert_grand_average_no_proj', '{subject_path}/fc-filt-{fmin}'
                                            '-{fmax}-Hz-{tmin}-{tmax}-s'
                                            '-no_proj-z_contrasts-ave.fif')
fname.add('hilbert_grand_average_proj', '{subject_path}/fc-filt-{fmin}'
                                            '-{fmax}-Hz-{tmin}-{tmax}-s'
                                            '-proj-z_contrasts-ave.fif')
fname.add('hilbert_statistics_proj', '{subject_path}/statistics/fc-filt-{fmin}'
                                    '-{fmax}-Hz-{tmin}-{tmax}-s'
                                    '-proj-z_contrasts-'
                                    '{first_event}-versus-{second_event}-stat-'
                                    '{stat_tmin}-{stat_tmax}-s-n_perm-{nperm}'
                                    '-seed-{seed}-pval-{pval}.npy')

## anatomy
fname.add('anatomy_simnibs_bem_surfaces',
          '{simnibs_bem_path}/{n_layers}-layers-bem.fif')
fname.add('anatomy_simnibs_bem_solutions',
          '{simnibs_bem_path}/{n_layers}-layers-bem-sol.fif')
fname.add('anatomy_volumetric_source_space', '{subject_bem_path}/volume'
                                            '-{spacing}_mm-src.fif')
fname.add('anatomy_simnibs_morph_volume', '{simnibs_bem_path}/volume-'
          '{spacing}_mm-morph.h5')
fname.add('anatomy_transformation', '{subject_path}/fc-trans.fif')
fname.add('anatomy_simnibs_forward_model', '{subject_path}/'
          'fc-simnibs-volume-{spacing}_mm-{n_layers}-layers-fwd.fif')

## source evoked
fname.add('source_evoked_beamformer_simnibs', '{subject_path}'
          '/beamformer_evoked/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}-'
          'simnibs-n_layers-{n_layers}-vl.stc')

fname.add('source_evoked_beamformer_simnibs_morph',
          '{subject_path}/beamformer_evoked/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}-'
          'simnibs-n_layers-{n_layers}-morph-vl-stc.h5')


fname.add('source_evoked_beamformer_contrast_simnibs', '{subject_path}'
          '/beamformer_evoked/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-'
          'reg-{reg}-contrast-{first_event}-versus-{second_event}-'
          'simnibs-n_layers-{n_layers}-{weight_norm}-vl.stc')


fname.add('source_evoked_beamformer_contrast_simnibs_morph', '{subject_path}'
          '/beamformer_evoked/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-'
          'reg-{reg}-contrast-{first_event}-versus-{second_event}-'
          'simnibs-n_layers-{n_layers}-{weight_norm}-morph-vl-stc.h5')

fname.add('source_evoked_beamformer_grand_average_simnibs', '{subject_path}/'
          'beamformer_evoked/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-{event}-'
          'filter-{first_event}-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}-morph-vl.h5')

fname.add('source_evoked_beamformer_statistics', '{subject_path}/statistics/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{first_event}-{second_event}-morph-stat-'
          '{stat_tmin}-{stat_tmax}-s-n_perm-{nperm}'
          '-seed-{seed}-condist-{condist}-pval-{pval}.npy')

## source hilbert

fname.add('source_hilbert_beamformer_simnibs', '{subject_path}/beamformer_hilbert/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}-vl.stc')

fname.add('source_hilbert_beamformer_simnibs_morph', '{subject_path}'
          '/beamformer_hilbert/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}-morph-vl-stc.h5')


fname.add('source_hilbert_beamformer_contrast_simnibs', '{subject_path}'
          '/beamformer_hilbert/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-'
          'reg-{reg}-contrast-{first_event}-versus-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}-vl.stc')


fname.add('source_hilbert_beamformer_contrast_simnibs_morph', '{subject_path}'
          '/beamformer_hilbert/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-'
          'reg-{reg}-contrast-{first_event}-versus-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}-morph-vl-stc.h5')

fname.add('source_hilbert_beamformer_grand_average_simnibs', '{subject_path}'
          '/beamformer_hilbert/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}-morph-vl.h5')


fname.add('source_hilbert_beamformer_contrast_grand_average_simnibs',
          '{subject_path}'
          '/beamformer_hilbert/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-'
          'reg-{reg}-contrast-{first_event}-versus-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}-morph-vl.h5')


## envelopes

fname.add('envelope_correlation', '{subject_envelope_path}/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-event-{event}-'
          'filter-{first_event}-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}.nc')
fname.add('envelope_correlation_morph_data', '{subject_envelope_path}/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-event-{event}-'
          'filter-{first_event}-{second_event}-{weight_norm}'
          '-simnibs-n_layers-{n_layers}-morph-data.npy')


#FIXME: check whether below needs weight norm and n-layers
fname.add('source_hilbert_beamformer_statistics', '{subject_path}/statistics/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{first_event}-{second_event}-morph-stat'
          '{stat_tmin}-{stat_tmax}-s-n_perm-{nperm}'
          '-seed-{seed}-condist-{condist}-pval-{pval}.npy')
fname.add('source_hilbert_beamformer_label', '{subject_path}'
          '/beamformer_hilbert/labels/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{label}-vl')
fname.add('source_hilbert_beamformer_contrast_label', '{subject_path}'
          '/beamformer_hilbert/labels/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s'
          '-reg-{reg}-contrast-{first_event}-versus-{second_event}-{label}-vl')
fname.add('source_hilbert_beamformer_label_grand_average', '{subject_path}'
          '/beamformer_hilbert/labels/'
          'fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s-reg-{reg}-'
          '{event}-filter-{first_event}-{second_event}-{label}-vl-stc.h5')
fname.add('source_hilbert_beamformer_contrast_label_grand_average',
          '{subject_path}'
          '/beamformer_hilbert/labels/fc-filt-{fmin}-{fmax}-Hz-{tmin}-{tmax}-s'
          '-reg-{reg}-contrast-{first_event}-versus-{second_event}-{label}-vl'
          '-stc.h5')

## figure names
fname.add('power_spectra_plot', '{subject_figure_path}/fc_power_spectra.png')
fname.add('events_plot', '{subject_figure_path}/fc_events.png')

## set the environment for FreeSurfer and MNE-Python
environ["SUBJECTS_DIR"] = fname.subjects_dir
