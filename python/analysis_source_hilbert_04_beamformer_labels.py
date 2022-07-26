#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 16:21:08 2022

@author: lau
"""

from config import (fname, submitting_method, atlas, rois, src_spacing,
                    hilbert_atlas_contrasts, hilbert_lcmv_regularization,
                    hilbert_tmin, hilbert_tmax, hilbert_fmins, hilbert_fmaxs)
from sys import argv
from helper_functions import should_we_run

import numpy as np
import mne
from nilearn import image
from os.path import exists


def this_function(subject, date, overwrite):
    src = mne.read_source_spaces(fname.anatomy_volumetric_source_space(
                                 subject='fsaverage', spacing=src_spacing))
    vertices = src[0]['vertno']
    for (fmin, fmax) in zip(hilbert_fmins, hilbert_fmaxs):
        for this_contrast in hilbert_atlas_contrasts:
            input_names = list()
            ## contrast 
            input_names.append(fname.source_hilbert_beamformer_contrast_morph(
                                            subject=subject,
                                            date=date,
                                            fmin=fmin,
                                            fmax=fmax,
                                            tmin=hilbert_tmin,
                                            tmax=hilbert_tmax,
                                            first_event=this_contrast[0],
                                            second_event=this_contrast[1],
                                            reg=hilbert_lcmv_regularization))
            ## first and second events
            for event in this_contrast:
                input_names.append(fname.source_hilbert_beamformer_morph(
                                    subject=subject,
                                    date=date,
                                    fmin=fmin,
                                    fmax=fmax,
                                    tmin=hilbert_tmin,
                                    tmax=hilbert_tmax,
                                    event=event,
                            reg=hilbert_lcmv_regularization,
                            first_event=this_contrast[0],
                            second_event=this_contrast[1])) # because h5
            
                
                
            labels = atlas['labels']
            for label in labels:
                if label not in rois:
                    continue ## skip to next
                output_names = list()
            ## contrast
                output_names.append(
                    fname.source_hilbert_beamformer_contrast_label(
                    subject=subject,
                    date=date,
                    fmin=fmin,
                    fmax=fmax,
                    tmin=hilbert_tmin,
                    tmax=hilbert_tmax,
                    first_event=this_contrast[0],
                    second_event=this_contrast[1],
                    reg=hilbert_lcmv_regularization,
                    label=label))
                ## first and second events
                for event in this_contrast:
                    output_names.append(
                        fname.source_hilbert_beamformer_label(
                        subject=subject,
                        date=date,
                        fmin=fmin,
                        fmax=fmax,
                        tmin=hilbert_tmin,
                        tmax=hilbert_tmax,
                        event=event,
                        reg=hilbert_lcmv_regularization,
                        first_event=this_contrast[0],
                        second_event=this_contrast[1],
                        label=label))
                    
                for name_index, output_name in enumerate(output_names):
                    if should_we_run(output_name + '-stc.h5', overwrite):
                        print(output_name + '-stc.h5')
                        input_name = input_names[name_index]
                        stc = mne.read_source_estimate(input_name)
                        ## save as nifti
                        filename = input_name + '.stc.h5.nii'
                        if not exists(filename):
                            stc.save_as_volume(filename, src, overwrite=True)
                        img = image.load_img(filename)
                        data = np.asanyarray(img.dataobj)
                        atlas_img = image.load_img(atlas['maps'])
                        ## interpolate atlas onto nifti
                        atlas_interpolated = \
                            image.resample_to_img(atlas_img, img, 'nearest')
                        atlas_interpolated_data = \
                            np.asanyarray(atlas_interpolated.dataobj)
                        ## find the label
                        label_index = \
                            int(atlas['indices'][atlas['labels'].index(label)])
                        ## create a mask
                        mask = atlas_interpolated_data == label_index
                        label_data = data[mask, :]
                        
                        ## create an stc with only the labels
                        stc_voxels = np.array(
                           np.unravel_index(vertices,
                                            img.shape[:3], order='F')).T
                        xs, ys, zs = np.where(mask > 0)

                        coordinates = np.concatenate((np.expand_dims(xs, 1),
                                                      np.expand_dims(ys, 1),
                                                      np.expand_dims(zs, 1)),
                                                      axis=1)

        
                        ## find the label vertices
                        label_vertices = list()
                        for coordinate_index, coordinate in enumerate(coordinates):
                            for voxel_index, voxel in enumerate(stc_voxels):
                                if np.all(voxel == coordinates[coordinate_index]):
                                    label_vertices.append(vertices[voxel_index])
                                    break

                        label_vertices.sort() ## works in place
                        
                        
                        ## create an stc
                        
                        full_data = np.zeros((len(vertices), len(stc.times)))
                        label_counter = 0
                        for vertex_index, vertex in enumerate(vertices):
                            if vertex in label_vertices:
                                full_data[vertex_index, :] = label_data[label_counter, :]
                                label_counter += 1
                                
                        
                                
                        stc_vol_full = mne.VolSourceEstimate(full_data,
                                                             [vertices],
                                                             stc.tmin,
                                                             stc.tstep)
                        
                        stc_vol_full.save(output_name, ftype='h5')
                        
                        
                        
if submitting_method == 'hyades_frontend':
    queue = 'highmem.q'
    job_name = 'hllcmv'
    n_jobs = 1
    deps = ['eve', 'hfilt', 'hepo', 'have', 'mri', 'ana', 'fwd', 'hlcmv',
            'mhlcmv']
    
if submitting_method == 'hyades_backend':
    print(argv[:])
    this_function(subject=argv[1], date=argv[2], overwrite=bool(int(argv[3])))