#!/bin/bash
#BSUB -W 24:00
#BSUB -o "/cluster/home/andresro/palm/output/Le_2A.%J.%I.txt"
#BSUB -e "/cluster/home/andresro/palm/output/Le_2A.%J.%I.txt"
#BSUB -J myjob[1-43]
##BSUB -w 'numended(58874163,*)'
#### BEGIN #####

search_path=(${PATH_DATA}/*.SAFE)
module load python_cpu/2.7.14 gdal/2.2.2

echo length ${#search_path[@]}
set -x

index=$((LSB_JOBINDEX-1))
file=${search_path[index]}
echo $file

python myScript.py $file  

#### END #####
