# file: tf3_4.py
# author: meikerwang
# forward and backward train

import tensorflow as tf
import numpy as np

BATCH_SIZE = 8
SEED = 23455

# 基于SEED产生随机数
rdm = np.random.RandomState(seed=SEED)
# 随机生成32个2维的0-1均匀分布的数据 作为输入数据集
X = rdm.rand(32, 2)
# 制作label数据集, label为 x0+x1 <1 为1否则为0
Y_ = [[int(x0 + x1 < 1)] for (x0, x1) in X]

print("X:\n", X)
print("Y_:\n", Y_)

# 1定义神经网络的输入、参数和输出,定义前向传播过程
x = tf.placeholder(dtype=tf.float32, shape=(None, 2))
y_ = tf.placeholder(dtype=tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal(shape=(2, 3), stddev=1.0, seed=1))
w2 = tf.Variable(tf.random_normal(shape=(3, 1), stddev=1.0, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# print(y)
# 定义损失函数和反向传播
loss_mse = tf.reduce_mean(tf.square(y - y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss=loss_mse)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    Loss = []
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    STEPS = 3000
    for i in range(STEPS):
        start = (i * BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        loss = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
        Loss.append(loss)
        if i % 500 == 0:
            print("After %d training step(s), loss_mse on all data is %g" % (i, loss))

    print("\n")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("loss:\n", Loss)
