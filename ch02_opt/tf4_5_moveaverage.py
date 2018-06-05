import tensorflow as tf

# 定义一个32位浮点变量，初始值为0.0  这个代码就是不断更新w1参数，优化w1参数，滑动平均做了个w1的影子
w1 = tf.Variable(0, dtype=tf.float32)

# 定义num_updates（NN的迭代轮数）,初始值为0，不可被优化（训练），这个参数不训练
global_steps = tf.Variable(0, trainable=False)

# 实例化滑动平均类，给衰减率为0.99，当前轮数global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, num_updates=global_steps)

# 在实际应用中会使用tf.trainable_variables()自动将所有待训练的参数汇总为列表
# 滑动平均节点
ema_op = ema.apply(tf.trainable_variables())

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 用ema.average(w1)获取w1滑动平均值 （要运行多个节点，作为列表中的元素列出，写在sess.run中）
    #       打印出当前参数w1和w1滑动平均值
    print("current global_step:", sess.run(global_steps))
    sess.run(ema_op)

    print("current w1:", sess.run([w1, ema.average(w1)]))
    print("current global_step:", sess.run(global_steps))

    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print("current global_step:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))

    sess.run(tf.assign(global_steps, 100))
    sess.run(tf.assign(w1, 10))

    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
    sess.run(ema_op)
    print("current global_steps:", sess.run(global_steps))
    print("current w1:", sess.run([w1, ema.average(w1)]))
