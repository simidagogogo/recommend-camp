# note: 线性回归

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sample_num = 100
batch_size = 10
epoch = 100

# Prepare train data
train_X = np.linspace(-1, 1, sample_num)
train_Y = 2 * train_X + .66 * np.random.randn(*train_X.shape) + 10  # *表示解压(脱去括号)，#(100,) -> 100,

train_X = np.reshape(train_X, (sample_num, 1))
train_Y = np.reshape(train_Y, (sample_num, 1))

# Define the model
X = tf.placeholder(tf.float32, shape=(batch_size, 1))
Y = tf.placeholder(tf.float32, shape=(batch_size, 1))
w = tf.Variable(tf.random.normal(shape=(1, 1)), name="weight", shape=(1, 1))
b = tf.Variable(tf.random.normal(shape=(1,)), name="bias", shape=(1,))

pred = tf.add(tf.multiply(X, w), b)
loss = tf.square(Y - pred)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
    # 将所有全局变量的初始化器汇总，并对其进行初始化
    sess.run(tf.initialize_all_variables())

    step = int(epoch * sample_num / batch_size)
    for i in range(step):
        indices = np.random.choice(sample_num, batch_size)
        _, w_value, b_value = sess.run([train_op, w, b],
                                       feed_dict={X: (train_X[indices]),
                                                  Y: (train_Y[indices])})
        if (i % 10 == 0):
            print("current step = {}, echo = {}".format(i, int(i / (sample_num / batch_size))))
            print("Epoch: {}, w: {}, b: {}".format(i, w_value, b_value))

plt.plot(train_X, train_Y, "+")
plt.plot(train_X, train_X.dot(w_value) + b_value)  # .dot()矩阵相乘
plt.show()
