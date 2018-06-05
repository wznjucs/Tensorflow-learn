# file: tf3_4.py
# author: meikerwang
# forward with placeholder [None, x_node]
import tensorflow as tf

# 用placeholder定义输入, 传入多组数据
x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal(shape=[3, 1], stddev=1.0, seed=1))

# forword
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    print("y in tf3_5.py is:\n", sess.run(fetches=y, feed_dict={x: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]}))
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

"""
y in tf3_5.py is:
 [[ 7.2020965]
 [17.270731 ]
 [27.339367 ]
 [37.408    ]]
w1:
 [[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
w2:
 [[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
"""
