# author: zhangda
# date:   2022/2/1 12:31
# note:   .

import tensorflow as tf


with tf.variable_scope("foo"):
    a = tf.get_variable("v", [1,])
    tf.get_variable_scope().reuse_variables()
    b = tf.get_variable("v", [1,])
    assert a == b

print(a)
print(b)
