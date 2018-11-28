#!/bin/bash
#BSUB -W 4:00
#BSUB -o /cluster/home/andresro/output/train_Le.%J.txt
#BSUB -e /cluster/home/andresro/output/train_Le.%J.txt
#BSUB -R "rusage[mem=1000,ngpus_excl_p=1]"
#BSUB -n 1
#BSUB -N
#BSUB -J TRAIN
#### BEGIN #####


module load python_gpu/2.7.14

python -c 'import tensorflow as tf; print(tf.__version__)'

#
#
#
#### END #####
