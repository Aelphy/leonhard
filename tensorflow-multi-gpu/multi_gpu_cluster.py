from single_gpu import *
from multi_gpu_single_process import *

cluster_specification = {
    "ps": ["localhost:2222"], # list of parameter servers,
    "worker": ["localhost:2223", "localhost:2224"] # list of workers
}


def start_parameter_server(task_index, cluster_specifcation):
    cluster_spec = tf.train.ClusterSpec(cluster_specification)
    server = tf.train.Server(cluster_spec, job_name='ps', task_index=task_index)
    server.join()
    
    
def start_worker(task_index, cluster_specification, dataset):  
    cluster_spec = tf.train.ClusterSpec(cluster_specification)
    server = tf.train.Server(cluster_spec, job_name="worker", task_index=task_index)
    
    worker_device = "/job:worker/task:{}".format(task_index)
    # `tf.train.replace_device_setter` automatically determines where to place variables
    with tf.device(tf.train.replica_device_setter(worker_device=worker_device,
                                                  cluster=cluster_spec)):
        iterator = dataset.make_one_shot_iterator()
        loss = training_model(lambda: iterator.get_next())
        
        optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)
        global_step = tf.train.get_or_create_global_step()
        update_op = optimizer.minimize(loss, global_step=global_step)
        
        # `tf.train.MonitoredTrainingSession` can be used as a drop-in replacement
        # for regular sessions.
        with tf.train.MonitoredTrainingSession(master=server.target,
                                               is_chief=task_index == 0) as sess:
            while not sess.should_stop():
                sess.run(update_op)
                
                

