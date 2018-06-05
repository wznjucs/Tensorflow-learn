import tensorflow as tf
from PIL import Image
import numpy as np
import mnist_forward
import mnist_backward


def pre_process(img_path):
    img = Image.open(img_path)
    re_image = img.resize((28, 28), Image.ANTIALIAS)  # ANTIALIAS 消除锯齿
    re_arr = np.array(re_image.convert("L"))
    threshold = 50
    for i in range(28):
        for j in range(28):
            re_arr[i][j] = 255 - re_arr[i][j]
            if re_arr[i][j] < threshold:
                re_arr[i][j] = 0
            else:
                re_arr[i][j] = 255
    nm_arr = re_arr.reshape([1, 28 * 28]).astype(np.float32)
    im_ready = np.multiply(nm_arr, 1.0 / 255)
    return im_ready


def restore_model(pic_arr):
    with tf.Graph().as_default():
        x = tf.placeholder(dtype=np.float32, shape=[None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)
        pre_y = tf.argmax(y, 1)
        ema = tf.train.ExponentialMovingAverage(
            decay=mnist_backward.MOVING_AVERAGE_DECAY
        )
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        with tf.Session() as sess:
            cpkt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if cpkt and cpkt.model_checkpoint_path:
                saver.restore(sess, cpkt.model_checkpoint_path)

                pre_value = sess.run(pre_y, feed_dict={x: pic_arr})
                return pre_value
            else:
                print("No checkpoint file found")
                return None


def application():
    test_num = int(input("input the number of test pictures: "))
    for i in range(test_num):
        test_path = input("input the path of test picture: ")
        test_pic_arr = pre_process(test_path)
        pre_value = restore_model(test_pic_arr)
        if pre_value is not None:
            print("The prediction number is:", pre_value)


if __name__ == "__main__":
    application()
