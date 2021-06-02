#! /bin/bash

import_MRI () {
    echo $1
    export SUBJECTS_DIR=/projects/MINDLAB2019_MEG-CerebellarClock/scratch/tactile_jitter/freesurfer
    raw_path='/projects/MINDLAB2019_MEG-CerebellarClock/raw/'
    temp=$1
    subject=${temp:0:4}
    cd ${raw_path}$1/files
    filename=$(ls | head -n 1) # gets first file
    recon-all -subjid $subject -i $filename
}

import_MRI $1
    

