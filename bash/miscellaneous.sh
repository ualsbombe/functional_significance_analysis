filecontent=( `cat "/home/lau/projects/functional_cerebellum/scripts/bash/all_MEG_paths.txt" `)
MEG_path=/home/lau/projects/functional_cerebellum/scratch/MEG
fs_path=/home/lau/projects/functional_cerebellum/scratch/freesurfer
simnibs_path=/home/lau/projects/functional_cerebellum/scratch/simnibs/


for path in "${filecontent[@]}"
do
    subject=${path:0:4}
    #cd $simnibs_path/$subject/fs_$subject/bem
    cd $MEG_path/$path/
    echo $subject
    ls fc-no-filt--0.75-0.75-s-tfr.h5
    #filenames=$(ls *None*)
   # old_string=None
  #  new_string=unit-gain
   # for filename in $filenames
   # do
    #    if [ ! -d $filename ]
     #   then
      #  new_name=${filename/$old_string/$new_string}
       # mv -v $filename $new_name
        #fi
    #done
    
done
   
