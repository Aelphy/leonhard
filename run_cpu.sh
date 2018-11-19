#!/bin/bash

module load python_gpu/3.6.0 # Avaliable versions are: 3.6.1 3.6.4 2.7.12 2.7.13 2.7.14
module load opencv # If you need it
# Amount of RAM per node - mem
# span[ptile=20] is nessesary for multi-node job - indicates the number of processors on each host that should be allocated
# -W time in hours which affects which queue will be selected
# -n - number of cores per node
bsub -n 1 -W 4:00 -R "rusage[mem=15096]" -R "span[ptile=10]" $PROGRAM_CMD
