# author: zhangda
# date: 2021/10/5 10:28
# note: linear_regression1.py

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt


class LinearRegression:
    # Define input data
    x_data = np.arange(100, step=.1)
    y_data = x_data + 20 * np.sin(x_data / 10)

    def __init__(self, n_samples=1000, batch_size=100, max_step=50000):
        # Define data size and batch size
        self.n_samples = n_samples
        self.batch_size = batch_size

        self.max_step = max_step
        self.epsilon = 1e-4
        self.log_step = 50
        self.last_loss_val = 0

        # Tensorflow is finicky about shapes, so resize
        self.x_data = np.reshape(self.x_data, (self.n_samples, 1))
        self.y_data = np.reshape(self.y_data, (self.n_samples, 1))

        # Define placeholders for input
        self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1), name="x")
        self.y = tf.placeholder(tf.float32, shape=(self.batch_size, 1), name="y")

    def train(self):
        # Define variables to be learned
        with tf.variable_scope("linear-regression"):
            # Note reuse=False, so these tensors are created a new
            # weight
            w = tf.get_variable("weights", (1, 1), dtype=tf.float32,
                                initializer=tf.random_normal_initializer())
            # bias
            b = tf.get_variable("bias", (1,), dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0))
            # predict value
            y_pred = tf.add(tf.matmul(self.x, w), b)

            # mse loss
            loss = tf.reduce_sum(tf.square(self.y - y_pred) / self.n_samples)  # 或者tf.reduce_mean(tf.square(y - y_pred))

        # Sample code to run full gradient descent:
        # Define optimizer operation
        optimize_op = tf.train.AdamOptimizer().minimize(loss)
        with tf.Session() as sess:
            # Initialize Variables in graph
            sess.run(tf.global_variables_initializer())

            # Gradient descent loop for 500 steps
            for step in range(self.max_step):
                # Select random minibatch
                indices = np.random.choice(self.n_samples, self.batch_size)

                # Do gradient descent step
                _, w_, b_, loss_val = sess.run([optimize_op, w, b, loss],
                                               feed_dict={self.x: self.x_data[indices],
                                                          self.y: self.y_data[indices]})
                if (abs(loss_val - self.last_loss_val) < self.epsilon):
                    print("early stop: current step = {}".format(step))
                    break
                self.last_loss_val = loss_val

                if (step % self.log_step == 0):
                    print("step = {}, W_ = {}, b_ = {}, loss_val = {}".format(step, w_, b_, loss_val))

            plt.plot(self.x_data, self.y_data)
            plt.plot(self.x_data, np.dot(self.x_data, w_) + b_)
            print(self.x_data.shape)
            print(self.y_data.shape)
            plt.show()

if __name__ == '__main__':
    lr_model = LinearRegression()
    lr_model.train()