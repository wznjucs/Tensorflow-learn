# file: tf3_4.py
# author: meikerwang
# forward with placeholder
import tensorflow as tf

# 定义输入placeholder和参数w1, w2
x = tf.placeholder(dtype=tf.float32, shape=[1, 2])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1.0, seed=1))

print(__file__)

# 定义forward过程: 两层fc
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 计算sess
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("y in tf3_4.py is:", sess.run(y, feed_dict={x: [[1.5, 2.0]]}))

"""
[[8.6353655]]
"""
