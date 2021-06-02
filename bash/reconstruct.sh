#! /bin/bash

reconstruct () {

    export SUBJECTS_DIR=/projects/MINDLAB2019_MEG-CerebellarClock/scratch/tactile_jitter/freesurfer
    temp=$1
    subject=${temp:0:4}
    recon-all -subjid $subject -all -openmp 2
}

reconstruct $1
