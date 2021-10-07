# 2018-07-24 10:55
# 2018-09-01 10:33 第二次阅读
# note: 线性回归

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

x_data = np.array([1, 2, 3, 4])
y_data = np.array([0, -1, -2, -3])

# 创建一个graph对象
graph = tf.Graph()
with graph.as_default():
    W = tf.Variable(0.3, dtype=tf.float32, name='weight')   # Variable必须初始化，可以随便赋一个初值
    b = tf.Variable(-0.3, dtype=tf.float32, name='bias')    # Variable 是全局变量
    x = tf.placeholder(tf.float32, name='x')                # placeholder为占位符变量，使用时一定要传值
    y = tf.placeholder(tf.float32, name='y')                # x和y都默认为一位数组

    predict = tf.add(tf.multiply(W, x), b)

    loss = tf.reduce_mean(tf.square(predict - y))

    # 使用梯度下降, GradientDescentOptimizer是优化器Optimizer的子类，创建梯度下降算法优化器的对象optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    # 调用minimize()方法，得到一个train操作对象
    train_op = optimizer.minimize(loss)

# 创建session运行计算图
with tf.Session(graph = graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        _, W_, b_, loss_ = sess.run([train_op, W, b, loss],
                                    feed_dict = {x: x_data, y: y_data})

        if i % 100 == 0:
            print("step = {}, W_ = {}, b_ = {}, loss_ = {}".format(i, W_, b_, loss_))

    print('train over')
    print(sess.run([W, b]))

    # plot
    plt.plot(x_data, y_data)
    plt.plot(x_data, np.dot(x_data, W_) + b_)
    plt.show()
