#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 10:08:00 2024

@author: lau
"""

#%% SUPPLEMENTARY FIGURE 1

from config import (fname, recordings, bad_subjects,
                    behavioural_data_time_stamps)
from manuscript_config import figure_path
from manuscript_helper_functions import (read_staircase_coefficients,
                                         inv_logit, set_rc_params,
                                         find_nearest)

from os.path import join
import numpy as np
import matplotlib.pyplot as plt


set_rc_params()

#%% LOAD DATA

group_intercept, group_slope, subject_intercepts, subject_slopes = \
    read_staircase_coefficients()
    
#%% TARGET CURRENTS

target_currents = np.load(join(fname.behavioural_path, 'target_currents.npy'))

#%% PLOT SIGMOIDS

fig = plt.figure()
x = np.arange(0, 10, 0.1)
subject_index = 0
for subject_intercept, subject_slope in \
    zip(subject_intercepts, subject_slopes):
    
   target_current = target_currents[subject_index] 
   y = inv_logit(group_intercept + subject_intercept + \
                 (group_slope + subject_slope) * x)
   plt.plot(x, y, '--', color='grey')
   plt.plot(target_current, y[find_nearest(x, target_current)], 'o',
            color='red')
   subject_index += 1
   

y_avg = inv_logit(group_intercept + group_slope * x)
plt.plot(x, y_avg, 'k', linewidth=3)
mean_current = 3.8 # 3.8 mA is the mean current across participants
plt.plot(mean_current, y_avg[find_nearest(x, mean_current)], 'ko',
         markersize=10)
plt.hlines(0.50, 0, 10, color='black', linestyles='dashed')
plt.xlabel('Current (mA)')
plt.ylabel('Proportion Correct')
plt.title('Estimated group and subject sigmoids')
plt.show()
fig.savefig(join(figure_path, 'sfig1'), dpi=300)