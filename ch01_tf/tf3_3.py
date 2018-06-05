# file: tf3_3.py
# author: meikerwang
# forward
import tensorflow as tf

# 定义输入和参数
x = tf.constant([[1.5, 2.0]])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0, seed=1))

# 定义forward过程: 两层fc
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 计算sess
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y in tf3_3.py is:", sess.run(y))

"""
[[8.6353655]]
"""
