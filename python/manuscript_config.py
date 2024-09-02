#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 16:02:05 2023

@author: lau
"""

#%% PATHS

figure_path = '/home/lau/Nextcloud/arbejde/AU/papers/' + \
              'functional_significance/figures/figure_parts'
         
## GENERAL              
        
rc_params = dict()
    
rc_params['font.size'] = 14
rc_params['font.weight'] = 'bold'
rc_params['lines.linewidth'] = 1.5

            
#%% FIG 1

#%% FIG 2

fig_2_settings = dict()
fig_2_settings['combinations'] = [
                dict(contrast=[
                    dict(event='s1', first_event='s1', second_event='s2'),
                    dict(event='s2', first_event='s1', second_event='s2'),
                        ]),
                dict(contrast=[
                    dict(event='w0', first_event='w0', second_event='w15'),
                    dict(event='w15', first_event='w0', second_event='w15'),
                        ]),
                dict(contrast=[
                    dict(event='o0', first_event='o0', second_event='o15'),
                    dict(event='o15', first_event='o0', second_event='o15'),
                        ]),
                ]
fig_2_settings['n_layers'] = 1
fig_2_settings['weight_norm'] = 'unit-noise-gain-invariant'
fig_2_settings['label_dict'] = dict(
                SI= dict(label='Postcentral_L', atlas='AAL',
                          restrict_time_index=261),
                SII=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=351),
                CL6=dict(label='Cerebelum_6_L', atlas='AAL',
                          restrict_time_index=351),
                CL1=dict(label='Cerebelum_Crus1_L', atlas='AAL',
                          restrict_time_index=351),
                 
                 )

#%% FIG 3

fig_3_settings = dict()
fig_3_settings['n_layers'] = 1
fig_3_settings['weight_norm'] = 'unit-noise-gain-invariant'
fig_3_settings['combination'] = dict(event='w0', first_event='w0',
                                     second_event='w15')
fig_3_settings['label_dict'] = dict(
                SII=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=351),
                CL6=dict(label='Cerebelum_6_L', atlas='AAL',
                          restrict_time_index=351),
                CL1=dict(label='Cerebelum_Crus1_L', atlas='AAL',
                          restrict_time_index=351),
                 )

#%% FIG 4

fig_4_settings = dict()
fig_4_settings['n_layers'] = 1
fig_4_settings['weight_norm'] = 'unit-noise-gain-invariant'
fig_4_settings['fmin'] = 14
fig_4_settings['fmax'] = 30
fig_4_settings['stat_tmin'] = 0.000
fig_4_settings['stat_tmax'] = 0.200
fig_4_settings['combinations'] = [
                dict(contrast=[
                    dict(event='s1', first_event='s1', second_event='s2'),
                    dict(event='s2', first_event='s1', second_event='s2'),
                        ]),
                dict(contrast=[
                    dict(event='w0', first_event='w0', second_event='w15'),
                    dict(event='w15', first_event='w0', second_event='w15'),
                        ]),
                dict(contrast=[
                    dict(event='o0', first_event='o0', second_event='o15'),
                    dict(event='o15', first_event='o0', second_event='o15'),
                        ]),
                ]

fig_4_settings['label_dict'] = dict(
                SI= dict(label='Postcentral_L', atlas='AAL',
                          restrict_time_index=None),
                SII=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=None, harvard_side='L',
                          boolean_index=10),
                CL6=dict(label='Cerebelum_6_L', atlas='AAL',
                          restrict_time_index=None),
                CL1=dict(label='Cerebelum_Crus1_L', atlas='AAL',
                          restrict_time_index=None),
                CL6_R=dict(label='Cerebelum_6_R', atlas='AAL',
                          restrict_time_index=None),
                 )


#%% FIG 5


fig_5_settings = dict()
fig_5_settings['n_layers'] = 1
fig_5_settings['weight_norm'] = 'unit-noise-gain-invariant'
fig_5_settings['fmin'] = 14
fig_5_settings['fmax'] = 30
fig_5_settings['stat_tmin'] = 0.000
fig_5_settings['stat_tmax'] = 0.200
fig_5_settings['combinations'] = [
                dict(contrast=[
                    dict(event='s1', first_event='s1', second_event='s2'),
                    dict(event='s2', first_event='s1', second_event='s2'),
                        ]),
                dict(contrast=[
                    dict(event='w0', first_event='w0', second_event='w15'),
                    dict(event='w15', first_event='w0', second_event='w15'),
                        ]),
                dict(contrast=[
                    dict(event='o0', first_event='o0', second_event='o15'),
                    dict(event='o15', first_event='o0', second_event='o15'),
                        ]),
                ]

fig_5_settings['label_dict'] = dict(
                SI= dict(label='Postcentral_L', atlas='AAL',
                          restrict_time_index=None),
                SII=dict(label='Parietal Operculum Cortex', atlas='harvard',
                          restrict_time_index=None, harvard_side='L',
                          boolean_index=10),
                TH=dict(label='Thalamus_L', atlas='AAL',
                          restrict_time_index=None),
                TH_R=dict(label='Thalamus_R', atlas='AAL',
                          restrict_time_index=None),
                 )


#%% FIG 6

fig_6_settings = dict()
fig_6_settings['n_layers'] = 1
fig_6_settings['weight_norm'] = 'unit-noise-gain-invariant'
fig_6_settings['fmin'] = 14
fig_6_settings['fmax'] = 30
fig_6_settings['tmin'] = -0.100
fig_6_settings['tmax'] =  0.100