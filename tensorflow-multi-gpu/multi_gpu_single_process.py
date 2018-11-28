from single_gpu import *

def parallel_training(model_fn, dataset):
    iterator = dataset.make_one_shot_iterator()
    
    def input_fn():
        with tf.device(None):
            # remove any device specifications for the input data
            return iterator.get_next()
    
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
    update_op, loss = create_parallel_optimization(model_fn, input_fn, optimizer)

    do_training(update_op, loss)
    
    
def device_example():
    # allocate variables on the CPU
    with tf.device('/cpu:0'):
        M = tf.get_variable('M', shape=[10,8], dtype=tf.float32)
        x = tf.get_variable('x', shape=[8, 1], dtype=tf.float32)
    # perform the operation on the fi=rst GPU device
    with tf.device('/gpu:0'):
        y = tf.matmul(M, x)
        
        
        
PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]

# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign


def device_example_2():
    # allocate variables on the CPU, perform the operation on the first GPU device
    with tf.device(assign_to_device('/gpu:0', '/cpu:0')):
        M = tf.get_variable('M', shape=[10,8], dtype=tf.float32)
        x = tf.get_variable('x', shape=[8, 1], dtype=tf.float32)
        y = tf.matmul(M, x)
        
        
def device_options():
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # your code here
        pass
    
    
def variable_scope_example():
    with tf.variable_scope('test_scope'):
        M = tf.get_variable('matrix1', shape=[10, 8], dtype=tf.float32)
        x = tf.get_variable('matrix2', shape=[8, 1], dtype=tf.float32)
        y = tf.matmul(M, x)
    # Here, we are instructing TensorFlow to reuse to variables declared above.
    # The `M` from above and the `N` from below reference the same Tensor!
    with tf.variable_scope('test_scope', reuse=True):
        N = tf.get_variable('matrix1', shape=[10, 8], dtype=tf.float32)
        z = tf.get_variable('matrix2', shape=[8, 1], dtype=tf.float32)
        w = tf.matmul(N, z)
        
        
def variable_scope_layers_failure():
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 10, 10, 1])
    with tf.variable_scope('test_scope'):
        tf.layers.conv2d(input_tensor, filters=10, kernel_size=[5, 5])

    # This will NOT reuse variables, since both `conv2d` implicitly create a variable scope with a
    # fresh name. Add `name='layer_name'` to make it work.
    with tf.variable_scope('test_scope', reuse=True):
        tf.layers.conv2d(input_tensor, filters=10, kernel_size=[5, 5])
        
        
def variable_scope_layers_correct():
    input_tensor = tf.placeholder(dtype=tf.float32, shape=[1, 10, 10, 1])
    with tf.variable_scope('test_scope'):
        tf.layers.conv2d(input_tensor, filters=10, kernel_size=[5, 5],
                         name='my_conv')
    # This does what it is supposed to do since both convolutional layers have been given
    # the same name
    with tf.variable_scope('test_scope', reuse=True):
        tf.layers.conv2d(input_tensor, filters=10, kernel_size=[5, 5],
                         name='my_conv')
        
        
def variable_scope_example_2():
    with tf.variable_scope('test_scope') as vscope:
        M = tf.get_variable('matrix1', shape=[10, 8], dtype=tf.float32)
        x = tf.get_variable('matrix2', shape=[8, 1], dtype=tf.float32)
        y = tf.matmul(M, x)
        
        vscope.reuse_variables()
        
        # variables are reused here
        N = tf.get_variable('matrix1', shape=[10, 8], dtype=tf.float32)
        z = tf.get_variable('matrix2', shape=[8, 1], dtype=tf.float32)
        w = tf.matmul(N, z)
        
        
def create_parallel_optimization(model_fn, input_fn, optimizer, controller="/cpu:0"):
    # This function is defined below; it returns a list of device ids like
    # `['/gpu:0', '/gpu:1']`
    devices = get_available_gpus()
        
    # This list keeps track of the gradients per tower and the losses
    tower_grads = []
    losses = []
    
    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):
                
                # Compute loss and gradients, but don't apply them yet
                loss = model_fn(input_fn)
                
                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)
                    
                losses.append(loss)
            
            # After the first iteration, we want to reuse the variables.
            outer_scope.reuse_variables()
                
    # Apply the gradients on the controlling device
    with tf.name_scope("apply_gradients"), tf.device(controller):
        # Note that what we are doing here mathematically is equivalent to returning the
        # average loss over the towers and compute the gradients relative to that.
        # Unfortunately, this would place all gradient-computations on one device, which is
        # why we had to compute the gradients above per tower and need to average them here.
        
        # This function is defined below; it takes the list of (gradient, variable) lists
        # and turns it into a single (gradient, variables) list.
        gradients = average_gradients(tower_grads)
        global_step = tf.train.get_or_create_global_step()
        apply_gradient_op = optimizer.apply_gradients(gradients, global_step)
        avg_loss = tf.reduce_mean(losses)

    return apply_gradient_op, avg_loss
        
        
# see https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

# see
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


if __name__ == '__main__':
    tf.reset_default_graph()
    parallel_training(training_model, training_dataset(epochs=2))
