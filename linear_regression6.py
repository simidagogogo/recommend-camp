# author: 
# date: 2021/10/5 19:39
# note:

import matplotlib.pyplot as plt
import numpy
import tensorflow as tf

learning_rate = 0.01
training_epochs = 1000
display_step = 50

train_X = numpy.linspace(0, 100, 50)
train_Y = numpy.multiply(train_X, 0.3) + 3 + 5 * numpy.random.randn(*train_X.shape)
plt.scatter(train_X, train_Y)
plt.show()

n_samples = train_X.shape[0]

X = tf.placeholder("float", name='X')
Y = tf.placeholder("float", name='Y')
W = tf.Variable(numpy.random.randn(), name="weight")
b = tf.Variable(numpy.random.randn(), name="bias")

pred = tf.add(tf.multiply(X, W), b)
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples) # MSE
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            _, W_, b_, cost_ = sess.run([optimizer, W, b, cost], feed_dict={X: x, Y: y})
        if epoch % 100 == 0:
            print("epoch = {}, w = {}, b = {}, cost = {}".format(epoch, W_, b_, cost_))

    print("优化完成!")
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, W_ * train_X + b_, label='Fitted line')
    plt.legend()
    plt.show()
