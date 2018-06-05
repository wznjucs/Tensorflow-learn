import tensorflow as tf
import time
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import lenet_forward
import lenet_backward

TEST_INTERNAL_SECS = 5


def test(mnist):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[
            mnist.test.num_examples,
            lenet_forward.IMAGE_SIZE,
            lenet_forward.IMAGE_SIZE,
            lenet_forward.NUM_CHANNELS
        ])
        y_ = tf.placeholder(tf.float32, shape=[None, lenet_forward.OUTPUT_SIZE])
        y = lenet_forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(
            decay=lenet_backward.MOVING_AVERAGE_DECAY
        )
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(lenet_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_steps = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

                    reshaped_x = np.reshape(mnist.test.images, (
                        mnist.test.num_examples,
                        lenet_forward.IMAGE_SIZE,
                        lenet_forward.IMAGE_SIZE,
                        lenet_forward.NUM_CHANNELS))

                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_x, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_steps, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERNAL_SECS)


if __name__ == "__main__":
    mnist = input_data.read_data_sets("./data/", one_hot=True)
    test(mnist)
