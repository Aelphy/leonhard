#!/bin/bash

module load python_cpu/3.6.0 # Avaliable versions are: 3.6.1 3.6.4 2.7.12 2.7.13 2.7.14
module load opencv # If you need it
# Availiable gpus GeForceGTX1080, GeForceGTX1080Ti
# Number of gpus per node ngpus_excl_p
# Amount of RAM per node - mem
# The GPU nodes have 20 cores, 8 GPUs, and 256 GB of RAM (of which only about 210 GB is usable)
# span[ptile=20] is nessesary for multi-node job
# -W time in hours which affects which queue will be selected
# -n - number of cores per node
bsub -n 1 -W 100:00 -R "rusage[mem=8092, ngpus_excl_p=1] select[gpu_model1==GeForceGTX1080] span[ptile=10]" $PROGRAM_CMD
