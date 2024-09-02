#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:58:04 2023

@author: lau
"""


from config import recordings, bad_subjects, fname
from manuscript_helper_functions import (create_comparison, get_max_vertices,
                                         get_peak_differences,
                                         read_subject_slopes_csv,
                                         read_glmer_coefficients, inv_logit,
                                         set_rc_params, roi_rewriting)
from manuscript_config import fig_3_settings, figure_path
import mne
from scipy.stats import spearmanr, pearsonr
import numpy as np
import matplotlib.pyplot as plt
from os.path import join

#%% FIGURE 3 - behaviour and correlation with cerebellar evoked responses

# behavior created in R

set_rc_params()

intercept, stim_type, slope, slope_change = read_glmer_coefficients()
max_variance = 0.10 # gotten from R

x = np.arange(0, max_variance, 0.001)
y1 = inv_logit(intercept + slope * x)
y2 = inv_logit(intercept + stim_type + (slope + slope_change) * x)

fig = plt.figure()
fig.set_size_inches(8, 6)
plt.plot(x, y1, color='#bcbd22')
plt.plot(x, y2, color='#2ca02c')
plt.ylim(0.70, 1.00)
plt.title('Behavioural performance')
plt.xlabel('Variance of last three stimulations (s²)')
plt.ylabel('Proportion correct')
plt.legend(['Omission', 'Weak'])
plt.show()

fig.savefig(join(figure_path, 'fig3_behaviour.png'), dpi=300)

# correlation

#%% get vertices

max_vertices_dict = dict()
src_path = fname.anatomy_volumetric_source_space(subject='fsaverage', 
                                                     spacing=7.5)
src = mne.read_source_spaces(src_path)

max_vertices_1, max_vertices_2 = get_max_vertices('fsaverage',
                                              '20210825_000000',
                      fig_3_settings['combination']['event'],
                      fig_3_settings['combination']['first_event'],
                      fig_3_settings['combination']['second_event'],
                      fig_3_settings['label_dict'], src)


#%% create comparisons


def get_correlations(rois, time_indices, conditions):

    diffs = dict()
    subject_slopes = dict()
    correlations = dict()

    for roi, time_index in zip(rois, time_indices):
        roi_dict = dict(vertex_index=[max_vertices_1, max_vertices_2],
                        time_index=time_index)
    
        comparison = create_comparison(roi_dict, 
                                       fig_3_settings['weight_norm'],
                                       'w0_w15')
        diffs[roi] = dict()
        diffs[roi]['data'] = get_peak_differences(recordings, bad_subjects,
                                                  comparison, roi)
        diffs[roi]['time_ms'] = time_index - 200 # FIXME: doesn't work for list
        subject_slopes[roi] = dict()
        subject_slopes[roi][conditions[0]], subject_slopes[roi][conditions[1]] = \
                        read_subject_slopes_csv(bad_subjects)
            
        correlations[roi] = dict()
        for condition in conditions:
            correlations[roi][condition] = dict()
            for correlation in ['spearman', 'pearson']:
                correlations[roi][condition][correlation] = dict()
                if correlation == 'spearman':
                    correlations[roi][condition][correlation] = \
                                               spearmanr(diffs[roi]['data'],
                                               subject_slopes[roi][condition],
                                                   nan_policy='omit')
                if correlation == 'pearson':
                    these_diffs = diffs[roi]['data'] 
                    these_slopes = np.array(subject_slopes[roi][condition])
                    correlations[roi][condition][correlation] = \
                        pearsonr(these_diffs[~np.isnan(these_diffs)],
                         these_slopes[~np.isnan(these_slopes)])
                        
    return correlations, subject_slopes, diffs


correlations, subject_slopes, diffs = get_correlations(['CL1', 'CL6', 'SII'],
                                                       [297, 297, 288],
                                                       ['weak', 'omissions'])


#%% plot comparisons

def plot_correlations(correlations, subject_slopes, diffs, roi, condition,
                      correlation_type, save=True):
    set_rc_params()
    these_slopes = subject_slopes[roi][condition]
    this_diff = diffs[roi]['data']
    fig = plt.figure()
    fig.set_size_inches(8, 6)
    plt.plot(these_slopes, this_diff, 'bo')
    plt.xlabel('Participant slopes (logistic regression)')
    ylabel = 'Difference in source strength:\n' +  condition.capitalize() + \
        ': no-jitter minus jitter (unit-noise-gain (T))'
    plt.ylabel(ylabel)
    title = 'Brain-behaviour correlation:\n' + condition.capitalize() + \
        ' evoked ' + roi_rewriting(roi) + ': ' + str(diffs[roi]['time_ms']) + \
            ' ms'
    plt.title(title)
    
    if correlation_type == 'pearson':
       X = np.array(these_slopes)
       X = X[~np.isnan(X)]
       X = np.expand_dims(X, 1)
       X = np.concatenate((np.expand_dims(np.ones(len(X)).T, 1), X), axis=1)
       y = np.expand_dims(this_diff[~np.isnan(this_diff)], axis=1)

       beta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

       def abline(slope, intercept):
           axes = plt.gca()
           x_vals = np.array(axes.get_xlim())
           y_vals = intercept + slope * x_vals
           plt.plot(x_vals, y_vals, 'b--')
           plt.show()
           
       abline(beta[1], beta[0])
       plt.legend(['Subjects', 'linear regression']) 
    
    rho, p = correlations[roi][condition][correlation_type]
    text = correlation_type.capitalize() + ' correlation:\n' + r'$\rho = $' + \
           str(np.round(rho, 3)) + '\n$\it{p}$ = ' + str(np.round(p, 4))
        
    
    x_text = np.nanmin(these_slopes) + 1.5
    y_text = np.nanmax(this_diff) - 1e-14
    plt.text(x_text, y_text, text, horizontalalignment='center')
    plt.show()
    if save:
        fig.savefig(join(figure_path, 'fig3_' + roi + ' ' + condition + \
                         ' ' + correlation_type + '.png'), dpi=300)


plot_correlations(correlations, subject_slopes, diffs, 'CL6', 'weak',
                  'spearman')
plot_correlations(correlations, subject_slopes, diffs, 'CL6', 'weak',
                  'pearson')
    
plot_correlations(correlations, subject_slopes, diffs, 'CL1', 'weak',
                  'spearman')

plot_correlations(correlations, subject_slopes, diffs, 'CL1', 'weak',
                  'pearson')

plot_correlations(correlations, subject_slopes, diffs, 'SII', 'weak',
                  'spearman', save=False)