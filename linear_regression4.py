# 2018-07-24 22:37

import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

x_data = np.float32(np.random.rand(2, 100)) # (2, 100)
y_data = np.dot([[0.100, 0.200]], x_data) + 0.300 # (1, 100)
plt.scatter(x_data[0, :], x_data[1, :])

X = tf.placeholder(tf.float32, [2, 100], name='x')
Y = tf.placeholder(tf.float32, [1, 100], name='y')
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0), tf.float32, name='W')
b = tf.Variable(0.0, tf.float32, name='b')

y = tf.matmul(W, X) + b
loss = tf.reduce_mean(tf.square(y - y_data))
                      
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)

    for i in range(1000):
        _, w_value, b_value = sess.run([train_op, w, b], 
                                       feed_dict = {X: x_data, Y: y_data})
        if i % 100 == 0:
            print("Step: {}, W: {}, b: {}".format(i, sess.run(W), sess.run(b)))
            
