from single_gpu import *
from multi_gpu_single_process import *
from multi_gpu_cluster import *

start_worker(0, cluster_specification, training_dataset())
