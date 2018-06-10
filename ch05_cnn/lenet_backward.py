import tensorflow as tf
import lenet_forward
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99

MODEL_SAVE_PATH = "./model/"
MODEL_NAME = "lenet_mnist_model"


def backward(mnist):
    x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, lenet_forward.INPUT_NODE])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, lenet_forward.OUTPUT_SIZE])
    y = lenet_forward.forward(x, train=True, regularizer=REGULARIZER)

    global_steps = tf.Variable(0, trainable=False)
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    loss = tf.reduce_mean(ce)
    loss_total = loss + tf.add_n(tf.get_collection("losses"))

    learning_rate = tf.train.exponential_decay(
        learning_rate=LEARNING_RATE_BASE,
        global_step=global_steps,
        decay_steps=mnist.train.num_examples / BATCH_SIZE,
        decay_rate=LEARNING_RATE_DECAY,
        staircase=True,
    )
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_total, global_step=global_steps)
    ema = tf.train.ExponentialMovingAverage(
        decay=MOVING_AVERAGE_DECAY,
        num_updates=global_steps
    )
    ema_op = ema.apply(tf.trainable_variables())

    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name="train")

    saver = tf.train.Saver()

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        for i in range(STEPS):
            batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_val, step_val = sess.run(
                [train_op, loss_total, global_steps], feed_dict={x: batch_xs, y_: batch_ys}
            )

            if i % 100 == 0:
                print("After %d training step(s), loss on the training batch is %g" % (step_val, loss_val))
                saver.save(sess, save_path=os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_steps)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    backward(mnist)


if __name__ == "__main__":
    main()
