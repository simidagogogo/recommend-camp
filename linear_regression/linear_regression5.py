# author: 
# date: 2021/10/5 19:39
# note:

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

learning_rate = 0.01
epochs = 1000
display_step = 50

train_X = numpy.linspace(0, 100, 50)
train_Y = numpy.multiply(train_X, 0.3) + 5 * numpy.random.randn(*train_X.shape)
plt.scatter(train_X, train_Y)
plt.show()

n_samples = train_X.shape[0]

X = tf.placeholder("float", name='X')
Y = tf.placeholder("float", name='Y')
W = tf.Variable(numpy.random.randn(), name="weight")
b = tf.Variable(numpy.random.randn(), name="bias")

predict = tf.add(tf.multiply(X, W), b)
# MSE, 或者tf.reduce_mean(tf.pow(pred - Y, 2)) / 2（写法很多）
loss = tf.reduce_sum(tf.pow(predict - Y, 2)) / (2 * n_samples)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for epoch in range(epochs):
        for (x, y) in zip(train_X, train_Y):
            _, W_, b_, cost_ = sess.run([optimizer, W, b, loss],
                                        feed_dict={X: x, Y: y})
        if epoch % display_step == 0:
            print("epoch = {}, w = {}, b = {}, cost = {}".format(epoch, W_, b_, cost_))

    print("优化完成!")
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, W_ * train_X + b_, label='Fitted line')
    plt.legend()
    plt.show()
