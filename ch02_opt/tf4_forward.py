import tensorflow as tf


def forward(x, regularizer):
    w1 = get_weight([2, 11], regularizer)
    b1 = get_bias([11])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([11, 1], regularizer)
    b2 = get_bias([1])
    y = tf.matmul(y1, w2) + b2
    return y


def get_weight(w_shape, regularizer):
    w = tf.Variable(tf.random_normal(shape=w_shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(b_shape):
    b = tf.Variable(tf.constant(value=0.01, shape=b_shape))
    return b
