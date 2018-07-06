# encoding:utf-8

import tensorflow as tf
import numpy as np
from lstm import LSTM


def weight_init(shape):
    initial = tf.random_uniform(shape, minval=-np.sqrt(5)*np.sqrt(1.0/shape[0]), maxval=np.sqrt(5)*np.sqrt(1.0/shape[0]))
    return tf.Variable(initial, trainable=True)


# 全部初始化成0
def zero_init(shape):
    initial = tf.Variable(tf.zeros(shape))
    return tf.Variable(initial, trainable=True)


# 正交矩阵初始化
def orthogonal_initializer(shape, scale=1.0):
    # https://github.com/Lasagne/Lasagne/blob/master/lasagne/init.py
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)  # this needs to be corrected to float32
    return tf.Variable(scale * q[:shape[0], :shape[1]], trainable=True, dtype=tf.float32)


def bias_init(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def shufflelists(data):
    ri = np.random.permutation(len(data))
    data = [data[i] for i in ri]
    return data


# 训练并记录
def train_epoch(EPOCH, train_data, test_data):
    for k in range(EPOCH):
        train0 = shufflelists(train_data)
        for i in range(len(train_data)):
            sess.run(train_step, feed_dict={inputs: train0[i][0], labels: train0[i][1]})
        tl = 0
        dl = 0
        for i in range(len(test_data)):
            dl += sess.run(loss, feed_dict={inputs: test_data[i][0], labels: test_data[i][1]})
        for i in range(len(train_data)):
            tl += sess.run(loss, feed_dict={inputs: train_data[i][0], labels: train_data[i][1]})
        print(k, 'train:', round(tl / 83, 3), 'test:', round(dl / 20, 3))


if __name__ == "__main__":
    D_input = 39
    D_label = 24
    learning_rate = 7e-5
    num_units = 1024
    EPOCH = 100
    train_data = ""
    test_data = " "
    # 样本的输入和标签
    inputs = tf.placeholder(tf.float32, [None, None, D_input], name="inputs")
    labels = tf.placeholder(tf.float32, [None, D_label], name="labels")
    # 实例LSTM类
    rnn_cell = LSTM(inputs, D_input, num_units, orthogonal_initializer)
    # 调用scan计算所有hidden states
    rnn0 = rnn_cell.all_steps()
    # 将3维tensor [n_steps, n_samples, D_cell]转成 矩阵[n_steps*n_samples, D_cell]
    # 用于计算outputs
    rnn = tf.reshape(rnn0, [-1, num_units])
    # 输出层的学习参数
    W = weight_init([num_units, D_label])
    b = bias_init([D_label])
    output = tf.matmul(rnn, W) + b
    # 损失
    loss = tf.reduce_mean((output-labels)**2)
    # 训练所需
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # 建立session并实际初始化所有参数
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    train_epoch(EPOCH, train_data, test_data)
