import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_DECAY_STEPS = 4  # DATA_SIZE / BATCH_SIZE 喂入多少轮BATCH_SIZE后，更新一次学习率

# 运行了几轮BATCH_SIZE的计数器，初值给0, 设为不被训练
global_steps = tf.Variable(0, trainable=False)

# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(
    learning_rate=LEARNING_RATE_BASE,
    global_step=global_steps,
    decay_steps=LEARNING_RATE_DECAY_STEPS,
    decay_rate=LEARNING_RATE_DECAY,
    staircase=True
)

w = tf.Variable(tf.constant(5.0))
loss_op = tf.square(w + 1)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_op, global_step=global_steps)
# train_FF = tf.train.AdadeltaOptimizer(learning_rate).minimize(loss_op, global_step=global_step)

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run([train_step, train_step])
        global_step_val = sess.run(global_steps)
        loss_val = sess.run(loss_op)
        w_val = sess.run(w)
        learning_rate_val = sess.run(learning_rate)
        print("After %s steps: global_step is %f, w is %f, learning rate is %f, loss is %f"
              % (i, global_step_val, w_val, learning_rate_val, loss_val)
              )
