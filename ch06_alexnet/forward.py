import tensorflow as tf

# 定义网络参数
n_input = 784  # 输入的维度
n_output = 10  # 标签的维度
learning_rate = 0.001
dropout = 0.75

# 存储所有的网络参数
weights = {
    # 使用截断的正态分布（标准差0.1）初始化卷积核的参数kernel，卷积核大小为3*3，channel为1，个数64
    'wc1': tf.Variable(tf.truncated_normal([3, 3, 1, 64], dtype=tf.float32, stddev=0.1), name='weights1'),
    'wc2': tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=0.1), name='weights2'),
    'wc3': tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=0.1), name='weights3'),
    'wc4': tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=0.1), name='weights4'),
    'wc5': tf.Variable(tf.truncated_normal([3, 3, 256, 128], dtype=tf.float32, stddev=0.1), name='weights5'),
    'wd1': tf.Variable(tf.truncated_normal([4 * 4 * 128, 1024], dtype=tf.float32, stddev=0.1), name='weights_fc1'),
    'wd2': tf.Variable(tf.random_normal([1024, 1024], dtype=tf.float32, stddev=0.1), name='weights_fc2'),
    'wd3': tf.Variable(tf.random_normal([1024, n_output], dtype=tf.float32, stddev=0.1), name='weights_output')
}
biases = {
    'bc1': tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases1'),
    'bc2': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases2'),
    'bc3': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases3'),
    'bc4': tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases4'),
    'bc5': tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases5'),
    'bd1': tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='biases_fc1'),
    'bd2': tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='biases_fc2'),
    'bd3': tf.Variable(tf.constant(0.0, shape=[n_output], dtype=tf.float32), trainable=True, name='biases_output')
}


# 定义函数print_activations来显示网络每一层结构，展示每一个卷积层或池化层输出tensor的尺寸
def print_activations(t):
    print(t.op.name, '', t.get_shape().as_list())


# 定义卷积操作
def conv2d(input, w, b):
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME")
    return tf.nn.relu(tf.nn.bias_add(conv, b))


# 定义池化操作
def max_pool(input):
    return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# 定义全连接操作
def fc(input, w, b):
    return tf.nn.relu(tf.add(tf.matmul(input, w), b))  # w*x+b，再通过非线性激活函数relu


# 定义网络结构
def alex_net(_input, _keep_prob):
    _input_r = tf.reshape(_input, [-1, 28, 28, 1])
    # 对图像做一个预处理，转换为tf支持的格式，即[n, h, w, c],-1是确定好其它3维后，让tf去推断剩下的1维

    with tf.name_scope('conv1'):
        _conv1 = conv2d(_input_r, weights['wc1'], biases['bc1'])
        print_activations(_conv1)  # 将这一层最后输出的tensor conv1的结构打印出来

    # # 这里参数基本都是AlexNet论文中的推荐值，但目前其他经典卷积神经网络模型基本都放弃了LRN（主要是效果不明显），
    # # 并且使用LRN也会让前馈、反馈的速度大大下降（整体速度降到1/3）
    # with tf.name_scope('_lrn1'):
    #     _lrn1 = tf.nn.lrn(_conv1, 4, bias=1.0, alpha=0.001/9, beta=0.75)

    with tf.name_scope('pool1'):
        _pool1 = max_pool(_conv1)
        print_activations(_pool1)

    with tf.name_scope('conv2'):
        _conv2 = conv2d(_pool1, weights['wc2'], biases['bc2'])
        print_activations(_conv2)

    # with tf.name_scope('_lrn2'):
    #     _lrn2 = tf.nn.lrn(_conv2, 4, bias=1.0, alpha=0.001/9, beta=0.75)

    with tf.name_scope('pool2'):
        _pool2 = max_pool(_conv2)
        print_activations(_pool2)

    with tf.name_scope('conv3'):
        _conv3 = conv2d(_pool2, weights['wc3'], biases['bc3'])
        print_activations(_conv3)

    with tf.name_scope('conv4'):
        _conv4 = conv2d(_conv3, weights['wc4'], biases['bc4'])
        print_activations(_conv4)

    with tf.name_scope('conv5'):
        _conv5 = conv2d(_conv4, weights['wc5'], biases['bc5'])
        print_activations(_conv5)

    with tf.name_scope('pool3'):
        _pool3 = max_pool(_conv5)
        print_activations(_pool3)
        nodes = weights["wd1"].get_shape().as_list()[0]
    print(nodes)
    _densel = tf.reshape(_pool3, [-1, nodes])
    # 定义全连接层的输入，把pool2的输出做一个reshape，变为向量的形式

    # pool_shape = _pool3.get_shape().as_list()
    # nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    with tf.name_scope('fc1'):
        _fc1 = fc(_densel, weights['wd1'], biases['bd1'])
        _fc1_drop = tf.nn.dropout(_fc1, _keep_prob)  # 为了减轻过拟合，使用Dropout层
        print_activations(_fc1_drop)

    with tf.name_scope('fc2'):
        _fc2 = fc(_fc1_drop, weights['wd2'], biases['bd2'])
        _fc2_drop = tf.nn.dropout(_fc2, _keep_prob)
        print_activations(_fc2_drop)

    with tf.name_scope('out'):
        _out = tf.add(tf.matmul(_fc2_drop, weights['wd3']), biases['bd3'])
        print_activations(_out)

    return _out
