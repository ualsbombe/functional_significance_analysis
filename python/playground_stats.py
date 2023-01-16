#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:36:55 2022

@author: lau
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
# mne.spatio_temporal_src_adjacency(src, n_times)
# mne.source_space.setup_volume_source_space(

# lala = data['o0_o15']['14_30_Hz']
lala = data['w0_w15']
# lala = lala[:, 1017, :] ## 446 CL1, 1017 CL6
lala = lala[:, vertex_indices['CL6'], :]
lala = np.mean(lala, 1)

mu = np.mean(lala, 0)
sem = np.std(lala, 0) / np.sqrt(26)


from scipy.stats import t
t_val = mu / sem 
p_val = (1 - t.cdf(np.abs(t_val), df=25)) * 2

times = np.arange(-0.200, 1.001, 0.001)
plt.figure()
plt.plot(times, t_val)
plt.plot(times, p_val, '-o')
plt.hlines(0.05, times[0], times[-1])
plt.show()


