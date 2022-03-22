#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 13:19:21 2021

@author: lau
"""

import mne
from config import fname
from helper_functions import read_split_raw

recordings = [
    dict(subject='0001', date='20210810_000000', mr_date=None),
    dict(subject='0002', date='20210804_000000', mr_date=None),
    dict(subject='0003', date='20210802_000000', mr_date='20210812_102146'),
    dict(subject='0004', date='20210728_000000', mr_date='20210811_164949'),
    dict(subject='0005', date='20210728_000000', mr_date='20210816_091907'),
    dict(subject='0006', date='20210728_000000', mr_date='20210811_173642'), # has a split raw file
    dict(subject='0007', date='20210728_000000', mr_date='20210812_105728'),
    dict(subject='0008', date='20210730_000000', mr_date='20210812_081520'),
    dict(subject='0009', date='20210730_000000', mr_date='20210812_141341'),
    dict(subject='0010', date='20210730_000000', mr_date='20210812_094201'),
    dict(subject='0011', date='20210730_000000', mr_date=None),
    dict(subject='0012', date='20210802_000000', mr_date='20210812_145235'),
    dict(subject='0013', date='20210802_000000', mr_date='20210811_084903'),
    dict(subject='0014', date='20210802_000000', mr_date='20210812_164859'),
    dict(subject='0015', date='20210804_000000', mr_date='20210811_133830'),
    dict(subject='0016', date='20210804_000000', mr_date='20210812_153043'),
    dict(subject='0017', date='20210805_000000', mr_date='20210820_123549'),
    dict(subject='0018', date='20210805_000000', mr_date='20210811_113632'),
    dict(subject='0019', date='20210805_000000', mr_date='20210811_101021'),
    dict(subject='0020', date='20210806_000000', mr_date='20210812_085148'),
    dict(subject='0021', date='20210806_000000', mr_date='20210811_145727'),
    dict(subject='0022', date='20210806_000000', mr_date='20210811_141117'),
    dict(subject='0023', date='20210809_000000', mr_date='20210812_112225'),
    dict(subject='0024', date='20210809_000000', mr_date='20210812_125146'),
    dict(subject='0026', date='20210810_000000', mr_date='20210811_120947'),
    dict(subject='0027', date='20210810_000000', mr_date='20210811_105000'),
    dict(subject='0028', date='20210817_000000', mr_date='20210820_111354'),
    dict(subject='0029', date='20210817_000000', mr_date='20210820_103315'),
    dict(subject='0030', date='20210817_000000', mr_date='20210820_085929'),
    dict(subject='0031', date='20210825_000000', mr_date='20210820_094714')
    ]

for recording in recordings:
    subject = recording['subject']
    date = recording['date']
    if subject != '0006':
        raw = mne.io.read_raw_fif(fname.raw_file(subject=subject, date=date))
    else:
        raw = read_split_raw(subject, date)
        
    raw.plot_psd(n_jobs=7)