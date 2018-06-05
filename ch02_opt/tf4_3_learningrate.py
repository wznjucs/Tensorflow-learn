import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

w = tf.Variable(tf.constant(5, dtype=tf.float32))

loss_op = tf.reduce_mean(tf.square(w + 1))
train_op = tf.train.GradientDescentOptimizer(0.2).minimize(loss_op)

L = []

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_op)
        loss_val = sess.run(loss_op)
        w_val = sess.run(w)
        L.append(loss_val)
        print("After %s steps: w is %f,   loss is %f." % (i, w_val, loss_val))
