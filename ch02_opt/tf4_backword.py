import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tf4_generateds
import tf4_forward

STEPS = 40000
BATCH_SIZE = 30
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.999
MOVING_AVERAGE_DECAY = 0.999
REGULARIZER = 0.01


def backward():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 2])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    X, Y_, Y_c = tf4_generateds.generateds()
    DATA_SIZE = tf4_generateds.DATA_SIZE

    y = tf4_forward.forward(x, REGULARIZER)

    global_steps = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_steps,
        decay_steps=DATA_SIZE / BATCH_SIZE,
        decay_rate=LEARNING_RATE_DECAY,
        staircase=True
    )

    # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # loss_entropy = tf.reduce_mean(ce)
    loss_mse = tf.reduce_mean(tf.square(y_ - y))
    loss_total = loss_mse = tf.add_n(tf.get_collection("losses"))

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total, global_step=global_steps)

    ema = tf.train.ExponentialMovingAverage(decay=MOVING_AVERAGE_DECAY, num_updates=global_steps)
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (i * BATCH_SIZE) % DATA_SIZE
            end = start + BATCH_SIZE
            sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start: end]})
            # print("global steps: %i, learning_rate: %g" % (sess.run(global_steps), sess.run(learning_rate)))
            if i % 1000 == 0:
                loss_val = sess.run(loss_total, feed_dict={x: X, y_: Y_})
                print("After %d steps, loss is: %f" % (i, loss_val))


if __name__ == "__main__":
    backward()
