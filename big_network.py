import numpy as np
import tensorflow as tf

inpt = tf.placeholder(tf.float32, [None, 640, 480, 3])

# GPU:0 computes L1(x)
with tf.device('/gpu:0'):
    conv11 = tf.layers.conv2d(
        inputs=inpt, filters=320, kernel_size=[5, 5],
        padding='same', activation=tf.nn.relu
    )

# GPU:1 computes L2(L1(X))
with tf.device('/gpu:1'):
    conv21 = tf.layers.conv2d(
        inputs=conv11, filters=32, kernel_size=[5, 5],
        padding='same', activation=tf.nn.relu
    )


sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(conv21, feed_dict={inpt: np.zeros((20, 640, 480, 3))})

