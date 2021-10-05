# 2018-07-24 20:05
#2018-09-01 10:42 第二次阅读

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Prepare train data
train_X = np.linspace(-1, 1, 100)
train_Y = 2 * train_X + 0.33 * np.random.randn(*train_X.shape) + 10  # *表示解压，#(100,) -> 100,

# Define the model
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w = tf.Variable(0.0, name = "weight")
b = tf.Variable(0.0, name = "bias")

loss = tf.square(Y - (X * w + b))
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# Create session to run
with tf.Session() as sess:
    # 将所有全局变量的初始化器汇总，并对其进行初始化
    init = tf.initialize_all_variables()
    sess.run(init)

    epoch = 10
    for i in range(epoch):
        for (x, y) in zip(train_X, train_Y):  # 一个x和一个y配对
            _, w_value, b_value = sess.run([train_op, w, b], feed_dict={X: x,Y: y})

        print("Epoch: {}, w: {}, b: {}".format(i+1, w_value, b_value))  # 每次迭代都输出一次

#draw
plt.plot(train_X, train_Y, "+")
plt.plot(train_X, train_X.dot(w_value) + b_value)   # .dot()矩阵相乘
plt.show()
