import tensorflow as tf

a = tf.constant([[2.0],[3.0]])
b = tf.constant([[4.0, 5.0]])

c = tf.matmul(a, b)

with tf.Session() as sess:
    result = sess.run(c)
    print(result)