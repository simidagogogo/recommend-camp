# author: zhangda
# date: 2021/10/5 10:28
# note: linear_regression1.py

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

# Define input data
x_data = np.arange(100, step=.1)
y_data = x_data + 20 * np.sin(x_data / 10)

# Define data size and batch size
n_samples = 1000
batch_size = 100

# Tensorflow is finicky about shapes, so resize
x_data = np.reshape(x_data, (n_samples, 1))
y_data = np.reshape(y_data, (n_samples, 1))

# Define placeholders for input
x = tf.placeholder(tf.float32, shape=(batch_size, 1), name="x")
y = tf.placeholder(tf.float32, shape=(batch_size, 1), name="y")

# Define variables to be learned
with tf.variable_scope("linear-regression"):
    # Note reuse=False, so these tensors are created a new
    # weight
    w = tf.get_variable("weights", (1, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer())
    # bias
    b = tf.get_variable("bias", (1, ), dtype=tf.float32,
                        initializer=tf.constant_initializer(0.0))
    # predict value
    y_pred = tf.add(tf.matmul(x, w), b)

    # mse loss
    loss = tf.reduce_sum(tf.square(y - y_pred) / n_samples) # 或者tf.reduce_mean(tf.square(y - y_pred))


# Sample code to run full gradient descent:
# Define optimizer operation
opt_operation = tf.train.AdamOptimizer().minimize(loss)

with tf.Session() as sess:
    # Initialize Variables in graph
    sess.run(tf.initialize_all_variables())

    # Gradient descent loop for 500 steps
    for step in range(5000):
        # Select random minibatch
        indices = np.random.choice(n_samples, batch_size)

        X_batch, y_batch = x_data[indices], y_data[indices]
        # Do gradient descent step
        _, W_, b_, loss_val = sess.run([opt_operation, w, b, loss],
                                       feed_dict={x: X_batch, y: y_batch})
        if (step % 50 == 0):
            print("step = {}, W_ = {}, b_ = {}, loss_val = {}".format(step, W_, b_, loss_val))

    plt.plot(x_data, y_data)
    plt.plot(x_data, np.dot(x_data, W_) + b_)
    print(x_data.shape)
    print(y_data.shape)
    plt.show()
