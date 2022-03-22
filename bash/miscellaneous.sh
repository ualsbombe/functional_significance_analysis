filecontent=( `cat "/home/lau/projects/functional_cerebellum/scripts/bash/all_MEG_paths.txt" `)
MEG_path=/home/lau/projects/functional_cerebellum/scratch/MEG


for path in "${filecontent[@]}"
do
    subject=${path:0:4}
    cd $MEG_path/$path/
    echo $subject
    ls -lq *z_con* | wc -l
    # ls *z_con*
done
   
