import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data

import mnist_forward
import mnist_backward

TEST_INTERNAL_SECS = 5


def test(mnist):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, shape=[None, mnist_forward.OUTPUT_NODE])
        y = mnist_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(
            decay=mnist_backward.MOVING_AVERAGE_DECAY
        )
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.arg_max(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_steps = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_steps, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return


if __name__ == "__main__":
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)
