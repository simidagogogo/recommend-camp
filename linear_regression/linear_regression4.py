# 2018-07-24 22:37
# note: 线性回归

import tensorflow as tf
import numpy as np

sample_nums = 1000
batch_size = 100

x_data = np.float32(np.random.rand(sample_nums, 2)) # (sample_nums, 2)
y_data = np.dot(x_data, [[0.100], [0.200]]) + 0.3 # (1, sample_nums)

X = tf.placeholder(tf.float32, shape=[None, 2], name='x')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), tf.float32, name='weight')
b = tf.Variable(0.0, tf.float32, name='bias')

y = tf.matmul(X, W) + b
loss = tf.reduce_mean(tf.square(tf.subtract(y, Y)))
                      
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    for i in range(sample_nums):
        indices = np.random.choice(sample_nums, batch_size)
        x_batch, y_batch = x_data[indices], y_data[indices]
        _, w_value, b_value, loss_ = sess.run([train_op, W, b, loss],
                                       feed_dict = {X: x_batch , Y: y_batch})
        if i % batch_size == 0:
            print("Step: {}, W: {}, b: {}, loss: {}".format(i, w_value, b_value, loss_))