#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:29:38 2021

@author: lau
"""

from config import fname, submitting_method
from os.path import isfile, join
from os import environ, listdir
from subprocess import Popen, PIPE, run
from sys import stdout
import mne
from time import time, sleep
import numpy as np
from scipy import stats as stats
import warnings

def should_we_run(save_path, overwrite, suppress_print=False,
                  print_file_to_be_saved=False):
    run_function = overwrite or not isfile(save_path)
    if not run_function:
        if not suppress_print:
            print('not overwriting: ' + save_path)
    if overwrite and isfile(save_path):
        print('Overwriting: ' + save_path)
    if print_file_to_be_saved and not overwrite:
        print('Saving: ' + save_path)
        
    return run_function

def run_process_and_write_output(command, subjects_dir, write_output=True):
    
    environment = environ.copy()
    if subjects_dir is not None:
        environment["SUBJECTS_DIR"] = subjects_dir
    process = Popen(command, stdout=PIPE,
                               env=environment)
    ## write bash output in python console
    if write_output:
        for c in iter(lambda: process.stdout.read(1), b''):
            stdout.write(c.decode('utf-8'))
        
def collapse_event_id(epochs, collapsed_event_id):
    for collapsed_event in collapsed_event_id:
        ## get rid of events that don't exist
        for old_event_id in \
            collapsed_event_id[collapsed_event]['old_event_ids']:
                if old_event_id not in epochs.event_id:
                    collapsed_event_id[collapsed_event]['old_event_ids'].pop()

        mne.epochs.combine_event_ids(epochs,
                     collapsed_event_id[collapsed_event]['old_event_ids'],
                     collapsed_event_id[collapsed_event]['new_event_id'],
                                         copy=False)
    return epochs


def check_if_all_events_present(events, event_id):
    existing_triggers = events[:, 2]
    keys_to_remove = list()
    for event in event_id:
        code = event_id[event]
        if not code in existing_triggers:
            keys_to_remove.append(event)
    for key in keys_to_remove:
        event_id.pop(key)
        
    return event_id

def read_split_raw(subject, date):
    first_file = fname.split_raw_file_1(subject=subject, date=date)
    second_file = fname.split_raw_file_2(subject=subject, date=date)
    raws = list()
    raws.append(mne.io.read_raw_fif(first_file, preload=True))
    raws.append(mne.io.read_raw_fif(second_file, preload=True))
    raw = mne.concatenate_raws(raws)
    
    return raw

def read_split_raw_info(subject, date):
    first_file = fname.split_raw_file_1(subject=subject, date=date)
    info = mne.io.read_info(first_file)
    
    return info

def find_common_channels(bad_channels):
    all_channels = list()
    for subject in bad_channels:
        for bad_channel in bad_channels[subject]:
            if 'MEG' in bad_channel:
                all_channels.append(bad_channel)
    unique_channels = np.unique(all_channels)
    return unique_channels            
    

def wait_submit(subject, deps, check_duration=10):
    if deps is not None:
        for dep in deps:
            process_running = True
            process_name = dep
            if subject != 'fsaverage':
                process_name += subject
            while process_running:
                output = run(['qstat', '-u', 'lau'], check=True, stdout=PIPE)
                processes = str(output.stdout)
                if process_name in processes:
                    message = 'Dependent process: ' + dep + ' of subject ' + \
                        subject + ' is still running'
                    print(message)
                    initial_time = time()
                    new_time = time()
                    while (new_time - check_duration) < initial_time:
                        new_time = time()
                else:
                    break 
                    
## links are removed and recreated (updating them)
## in the scripts/python/qsub library where they will be retrieved from
## when submitting the job in "submit_job" below
def update_links_in_qsub():
    python_path = fname.python_path
    qsub_path = fname.python_qsub_path
        
    script_names = listdir(python_path)
    for script_name in script_names:
        if 'analysis' in script_name:
            sleep(0.01)
            ## clean
            command = ['rm', join(qsub_path, script_name)]
            run_process_and_write_output(command, subjects_dir=None,
                                         write_output=False)
            ## create link
            sleep(0.01)
            command = ['ln', '-sf', join(python_path, script_name), 
                 join(qsub_path, script_name)]
            run_process_and_write_output(command, subjects_dir=None,
                                         write_output=False)
            
def submit_job(recording, function_name, overwrite=False, wait=True):
    ## make sure that there are symbolic links in the qsub folder, where
    ## the cluster calls functions from
    update_links_in_qsub()
    subject = recording['subject']
    date = recording['date']
    mr_date = recording['mr_date']
    
    if submitting_method == 'local':   
        if function_name == 'create_folders':
            from analysis_00_create_folders import create_folders
            create_folders(subject, date)

        import_statement = 'from ' + function_name + ' import this_function' 
        exec(import_statement, globals()) ## FIXME: is this dangerous?
        this_function(subject, date, overwrite)
    
    elif submitting_method == 'hyades_frontend':
        from stormdb.cluster import ClusterJob    
        overwrite_string = str(int(overwrite))
        
        import_statement = 'from ' + function_name + ' import ' + \
            'queue, job_name, n_jobs, deps'    
        exec(import_statement, globals())
        
        if function_name == \
            'analysis_anatomy_00_import_reconstruct_watershed_' + \
                'and_scalp_surfaces' \
            or function_name == 'analysis_anatomy_00_segmentation':
            date = mr_date
                
        
        if date is not None:
        
            cmd = "python " + function_name + '.py' + " " + \
                    subject + " " + date + " " + overwrite_string
          
            if wait:
                wait_submit(subject, deps)
                      
            cj = ClusterJob(
                cmd=cmd,
                queue=queue,
                job_name=job_name + subject,
                proj_name='MINDLAB2021_MEG-CerebellarClock-FuncSig',
                working_dir=fname.python_qsub_path,
                n_threads=n_jobs)
            cj.submit()
            
    else:
        print(submitting_method)
        raise RuntimeError('Unspecified "submitting_method"')
        
def wilcoxon(x, y=None, zero_method="wilcox", correction=False,
             alternative="two-sided"):
    """
    Adapted by Lau to spit out z-values
    Calculate the Wilcoxon signed-rank test.

    The Wilcoxon signed-rank test tests the null hypothesis that two
    related paired samples come from the same distribution. In particular,
    it tests whether the distribution of the differences x - y is symmetric
    about zero. It is a non-parametric version of the paired T-test.

    Parameters
    ----------
    x : array_like
        Either the first set of measurements (in which case `y` is the second
        set of measurements), or the differences between two sets of
        measurements (in which case `y` is not to be specified.)  Must be
        one-dimensional.
    y : array_like, optional
        Either the second set of measurements (if `x` is the first set of
        measurements), or not specified (if `x` is the differences between
        two sets of measurements.)  Must be one-dimensional.
    zero_method : {'pratt', 'wilcox', 'zsplit'}, optional
        The following options are available (default is 'wilcox'):
     
          * 'pratt': Includes zero-differences in the ranking process,
            but drops the ranks of the zeros, see [4]_, (more conservative).
          * 'wilcox': Discards all zero-differences, the default.
          * 'zsplit': Includes zero-differences in the ranking process and 
            split the zero rank between positive and negative ones.
    correction : bool, optional
        If True, apply continuity correction by adjusting the Wilcoxon rank
        statistic by 0.5 towards the mean value when computing the
        z-statistic.  Default is False.
    alternative : {"two-sided", "greater", "less"}, optional
        The alternative hypothesis to be tested, see Notes. Default is
        "two-sided".

    Returns
    -------
    statistic : float
        If `alternative` is "two-sided", the sum of the ranks of the
        differences above or below zero, whichever is smaller.
        Otherwise the sum of the ranks of the differences above zero.
    pvalue : float
        The p-value for the test depending on `alternative`.

    See Also
    --------
    kruskal, mannwhitneyu

    Notes
    -----
    The test has been introduced in [4]_. Given n independent samples
    (xi, yi) from a bivariate distribution (i.e. paired samples),
    it computes the differences di = xi - yi. One assumption of the test
    is that the differences are symmetric, see [2]_.
    The two-sided test has the null hypothesis that the median of the
    differences is zero against the alternative that it is different from
    zero. The one-sided test has the null hypothesis that the median is 
    positive against the alternative that it is negative 
    (``alternative == 'less'``), or vice versa (``alternative == 'greater.'``).

    The test uses a normal approximation to derive the p-value (if
    ``zero_method == 'pratt'``, the approximation is adjusted as in [5]_).
    A typical rule is to require that n > 20 ([2]_, p. 383). For smaller n,
    exact tables can be used to find critical values.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
    .. [2] Conover, W.J., Practical Nonparametric Statistics, 1971.
    .. [3] Pratt, J.W., Remarks on Zeros and Ties in the Wilcoxon Signed
       Rank Procedures, Journal of the American Statistical Association,
       Vol. 54, 1959, pp. 655-667. :doi:`10.1080/01621459.1959.10501526`
    .. [4] Wilcoxon, F., Individual Comparisons by Ranking Methods,
       Biometrics Bulletin, Vol. 1, 1945, pp. 80-83. :doi:`10.2307/3001968`
    .. [5] Cureton, E.E., The Normal Approximation to the Signed-Rank
       Sampling Distribution When Zero Differences are Present,
       Journal of the American Statistical Association, Vol. 62, 1967,
       pp. 1068-1069. :doi:`10.1080/01621459.1967.10500917`

    Examples
    --------
    In [4]_, the differences in height between cross- and self-fertilized
    corn plants is given as follows:

    >>> d = [6, 8, 14, 16, 23, 24, 28, 29, 41, -48, 49, 56, 60, -67, 75]

    Cross-fertilized plants appear to be be higher. To test the null
    hypothesis that there is no height difference, we can apply the
    two-sided test:

    >>> from scipy.stats import wilcoxon
    >>> w, p = wilcoxon(d)
    >>> w, p
    (24.0, 0.04088813291185591)

    Hence, we would reject the null hypothesis at a confidence level of 5%,
    concluding that there is a difference in height between the groups.
    To confirm that the median of the differences can be assumed to be
    positive, we use:

    >>> w, p = wilcoxon(d, alternative='greater')
    >>> w, p
    (96.0, 0.020444066455927955)

    This shows that the null hypothesis that the median is negative can be
    rejected at a confidence level of 5% in favor of the alternative that
    the median is greater than zero. The p-value based on the approximation
    is within the range of 0.019 and 0.054 given in [2]_.
    Note that the statistic changed to 96 in the one-sided case (the sum
    of ranks of positive differences) whereas it is 24 in the two-sided
    case (the minimum of sum of ranks above and below zero).

    """

    if zero_method not in ["wilcox", "pratt", "zsplit"]:
        raise ValueError("Zero method should be either 'wilcox' "
                         "or 'pratt' or 'zsplit'")

    if alternative not in ["two-sided", "less", "greater"]:
        raise ValueError("Alternative must be either 'two-sided', "
                         "'greater' or 'less'")

    if y is None:
        d = np.asarray(x)
        if d.ndim > 1:
            raise ValueError('Sample x must be one-dimensional.')
    else:
        x, y = map(np.asarray, (x, y))
        if x.ndim > 1 or y.ndim > 1:
            raise ValueError('Samples x and y must be one-dimensional.')
        if len(x) != len(y):
            raise ValueError('The samples x and y must have the same length.')
        d = x - y

    if zero_method in ["wilcox", "pratt"]:
        n_zero = np.sum(d == 0, axis=0)
        if n_zero == len(d):
            raise ValueError("zero_method 'wilcox' and 'pratt' do not work if "
                             "the x - y is zero for all elements.")

    if zero_method == "wilcox":
        # Keep all non-zero differences
        d = np.compress(np.not_equal(d, 0), d, axis=-1)

    count = len(d)
    if count < 10:
        warnings.warn("Sample size too small for normal approximation.")
        
    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r, axis=0) ## FIXME: real part is compared by np
    r_minus = np.sum((d < 0) * r, axis=0)

    if zero_method == "zsplit":
        r_zero = np.sum((d == 0) * r, axis=0)
        r_plus += r_zero / 2.
        r_minus += r_zero / 2.

    # return min for two-sided test, but r_plus for one-sided test
    # the literature is not consistent here
    # r_plus is more informative since r_plus + r_minus = count*(count+1)/2,
    # i.e. the sum of the ranks, so r_minus and the min can be inferred
    # (If alternative='pratt', r_plus + r_minus = count*(count+1)/2 - r_zero.)
    # [3] uses the r_plus for the one-sided test, keep min for two-sided test
    # to keep backwards compatibility
    if alternative == "two-sided":
        T = min(r_plus, r_minus)
    else:
        T = r_plus
    mn = count * (count + 1.) * 0.25
    se = count * (count + 1.) * (2. * count + 1.)

    if zero_method == "pratt":
        r = r[d != 0]
        # normal approximation needs to be adjusted, see Cureton (1967)
        mn -= n_zero * (n_zero + 1.) * 0.25
        se -= n_zero * (n_zero + 1.) * (2. * n_zero + 1.)

    replist, repnum = stats.find_repeats(r)
    if repnum.size != 0:
        # Correction for repeated elements.
        se -= 0.5 * (repnum * (repnum * repnum - 1)).sum()

    se = np.sqrt(se / 24)

    # apply continuity correction if applicable
    d = 0
    if correction:
        if alternative == "two-sided":
            d = 0.5 * np.sign(T - mn)
        elif alternative == "less":
            d = -0.5
        else:
            d = 0.5

    # compute statistic and p-value using normal approximation
    z = (T - mn - d) / se
    ## added by lau, necessary to get right sign of z, which is not 
    ## calculated in the scipy implementation
    if r_plus > r_minus:
        z = np.abs(z)

    return z        
        
            