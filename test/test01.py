import tensorflow as tf
from tensorflow.python.framework import ops

a = tf.random_uniform([1])
b = tf.random_normal([1])
print("Session 1")
with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'

print("Session 2")
with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A3'
    print(sess2.run(a))  # generates 'A4'
    print(sess2.run(b))  # generates 'B3'

a = tf.random_uniform([1], seed=1)
b = tf.random_normal([1])
# Repeatedly running this block with the same graph will generate the same
# sequence of values for 'a', but different sequences of values for 'b'.
print("Session 1")
with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'
print("Session 2")
with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B3'
    print(sess2.run(b))  # generates 'B4'

tf.set_random_seed(1234)
a = tf.random_uniform([1])
b = tf.random_normal([1])
# Repeatedly running this block with the same graph will generate the same
# sequences of 'a' and 'b'.
print("Session 1")
with tf.Session() as sess1:
    print(sess1.run(a))  # generates 'A1'
    print(sess1.run(a))  # generates 'A2'
    print(sess1.run(b))  # generates 'B1'
    print(sess1.run(b))  # generates 'B2'
print("Session 2")
with tf.Session() as sess2:
    print(sess2.run(a))  # generates 'A1'
    print(sess2.run(a))  # generates 'A2'
    print(sess2.run(b))  # generates 'B1'
    print(sess2.run(b))  # generates 'B2'


a = tf.get_variable()

A = {
    "w": tf.get_variable(name="conv2_w", shape=[2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=1)),
    "b": tf.Variable(tf.constant(0.0, shape=[16]), name="conv2_b"),
    "strides": [1, 1, 1, 1],
    "padding": "SAME",
},

a = tf.contrib.layers.xavier_initializer()

b = tf.layers