#!/bin/bash
#BSUB -W 24:00
#BSUB -o /cluster/home/andresro/palm/outputtrain_Le${object}.%J.%I.txt
#BSUB -e /cluster/home/andresro/palm/output/train_Le${object}.%J.%I.txt
#BSUB -R "rusage[mem=200000,ngpus_excl_p=1]"
#BSUB -n 1
#BSUB -N
#BSUB -J "tr_ar[1-4]"
##BSUB -w 'numended(21938,*)'
#### BEGIN #####


module load python_gpu/2.7.14


# Parameters of each script
parameters=(DL2,DL3,Simple,Simple_atrous)


index=$((LSB_JOBINDEX-1))
model=${parameters[index]}
echo $model


python2 -u train_reg.py \
    --model=$model
    
    
#
#
#
#### END #####
