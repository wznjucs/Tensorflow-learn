from datetime import datetime
import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward

mnist = input_data.read_data_sets('data/', one_hot=True)
print("MNIST READY")

# 上面神经网络结构定义好之后，下面定义一些超参数
num_epochs = 1  # 所有样本迭代 20 次
batch_size = 64  # 每次迭代的 batch size
learning_rate = 0.001
dropout = 0.75

X = tf.placeholder(tf.float32, [None, forward.n_input])  # 用placeholder先占地方，样本个数不确定为None
Y = tf.placeholder(tf.float32, [None, forward.n_output])  # 用placeholder先占地方，样本个数不确定为None
keep_prob = tf.placeholder(tf.float32)

# 前向传播的预测值
Y_pred = forward.alex_net(X, keep_prob)

# 交叉熵损失函数，参数分别为预测值 Y_pred 和实际label值 Y，reduce_mean 为求平均 loss
cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(logits=Y_pred, labels=tf.argmax(Y, 1))
)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#  tf.equal()对比预测值的索引和实际label的索引是否一样，一样返回True，不一样返回False
correct_pred = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
# 将pred即True或False转换为1或0,并对所有的判断结果求均值
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
costs = []
print("FUNCTIONS READY")

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)  # 在sess里run一下初始化操作

    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_cost = 0.
        num_batches = int(mnist.train.num_examples / batch_size)

        for one_batch in range(1):
            # Select a batch
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: dropout})
            epoch_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 1.0}) / num_batches

        costs.append(epoch_cost)
        train_accuracy = accuracy.eval({X: mnist.train.images, Y: mnist.train.labels, keep_prob: 1.0})
        test_accuracy = accuracy.eval({X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1.0})
        print("Epoch: %03d/%03d cost: %.9f TRAIN ACCURACY: %.6f TEST ACCURACY: %.6f" % (
            (epoch + 1), num_epochs, epoch_cost, train_accuracy, test_accuracy))

        duration = time.time() - start_time
        print('%s: epoch %d, duration = %.3f' % (datetime.now(), epoch + 1, duration))

print("DONE")
