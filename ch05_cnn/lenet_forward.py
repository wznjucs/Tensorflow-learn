import tensorflow as tf

IMAGE_SIZE = 28
NUM_CHANNELS = 1
CONV1_SIZE = 5
CONV1_KERNEL_NUM = 32
CONV2_SIZE = 5
CONV2_KERNEL_NUM = 64
FC_SIZE = 512
OUTPUT_SIZE = 10


def get_weight(w_shape, regularizer=None):
    w = tf.Variable(tf.truncated_normal(shape=w_shape, stddev=0.01))
    if regularizer != None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(b_shape):
    b = tf.Variable(tf.zeros(shape=b_shape))
    return b


def conv2d(x, W):
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')
    return conv


def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def forward(x, train=False, regularizer=None):
    conv1_w = get_weight([CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)

    conv2_w = get_weight([CONV1_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    conv2_b = get_bias([CONV2_KERNEL_NUM])
    conv2 = conv2d(pool1, conv2_w)
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    pool2 = max_pool_2x2(relu2)

    # 将pool2 从张量变为向量
    pool_shape = pool2.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_weight([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    if train:
        fc1 = tf.nn.dropout(fc1, keep_prob=0.5)

    fc2_w = get_weight([FC_SIZE, OUTPUT_SIZE], regularizer)
    fc2_b = get_bias([OUTPUT_SIZE])
    fc2 = tf.matmul(fc1, fc2_w) + fc2_b

    return fc2
