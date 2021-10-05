# 2018-07-24 22:37

import tensorflow as tf
import numpy as np

x_data = np.float32(np.random.rand(100, 2)) # (100, 2)
y_data = np.dot(x_data, [[0.100], [0.200]] ) + 0.300 # (100, 1), 预测结果应该是0.1和0.2

# Define the model
X = tf.placeholder(tf.float32, shape = [None, 2])
y = tf.placeholder(tf.float32, shape = [None, 1])
W = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0) )
b = tf.Variable(0.0)

y = tf.matmul(X, W) + b  #(None, 1)

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.05)
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    step = 2000
    for i in range(step):
        _, W_, b_ = sess.run([train_op, W, b], feed_dict = {X: x_data, Y: y_data})

        if i % 100 == 0:
            print("Step: {}, W: {}, b: {}".format(i, W_, b_))
            
            
