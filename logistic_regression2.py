# 2018-07-24 10:55
# 2018-09-01 10:33 第二次阅读
# 线性回归

import tensorflow as tf

# 创建一个graph对象
graph = tf.Graph()
with graph.as_default():
    W = tf.Variable(0.3, tf.float32)   # Variable必须初始化，可以随便赋一个初值
    b = tf.Variable(-0.3, tf.float32)  # Variable 是全局变量
    x = tf.placeholder(tf.float32)      # placeholder为占位符变量，使用时一定要传值
    y = tf.placeholder(tf.float32)      # x和y都默认为一位数组

    linear_model = W * x + b                    # 线性回归模型

    # 计算误差值平方的均值
    loss = tf.reduce_mean(tf.square(linear_model - y))

    # 使用梯度下降, GradientDescentOptimizer是优化器Optimizer的子类，创建梯度下降算法优化器的对象optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)

    # 对对象optimizer调用minimize()方法，得到一个train变量
    train = optimizer.minimize(loss)    

# 创建session运行计算图
with tf.Session(graph = graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(1000):
        sess.run([train], feed_dict = {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

        # 为了显示运行进度，每隔100次循环显示一次输出
        if i%100 == 0:
            print(sess.run([W, b]))

    print('train over')

    # 打印最后的训练结果
    print(sess.run([W, b]))
    print(sess.run(tf.shape(W)))
    print(sess.run(tf.rank(W)))
    
