# encoding:utf-8
import tensorflow as tf


class LSTM(object):
    def __init__(self, incoming, D_input, D_cell, initializer, f_bias=1.0):
            # incoming是用来接收输入数据的，其形状为[n_samples, n_steps, D_cell]
        self.incoming = incoming
        # 输入的维度
        self.D_input = D_input
        # LSTM的hidden state的维度，同时也是memory cell的维度
        self.D_cell = D_cell
        # parameters
        # 输入门的 三个参数
        # igate = W_xi.* x + W_hi.* h + b_i
        self.W_xi = initializer([self.D_input, self.D_cell])
        self.W_hi = initializer([self.D_cell, self.D_cell])
        self.b_i  = tf.Variable(tf.zeros([self.D_cell]))
        # 遗忘门的 三个参数
        # fgate = W_xf.* x + W_hf.* h + b_f
        self.W_xf = initializer([self.D_input, self.D_cell])
        self.W_hf = initializer([self.D_cell, self.D_cell])
        self.b_f  = tf.Variable(tf.constant(f_bias, shape=[self.D_cell]))
        # 输出门的 三个参数
        # ogate = W_xo.* x + W_ho.* h + b_o
        self.W_xo = initializer([self.D_input, self.D_cell])
        self.W_ho = initializer([self.D_cell, self.D_cell])
        self.b_o  = tf.Variable(tf.zeros([self.D_cell]))
        # 计算新信息的三个参数
        # cell = W_xc.* x + W_hc.* h + b_c
        self.W_xc = initializer([self.D_input, self.D_cell])
        self.W_hc = initializer([self.D_cell, self.D_cell])
        self.b_c  = tf.Variable(tf.zeros([self.D_cell]))

        # 最初时的hidden state和memory cell的值，二者的形状都是[n_samples, D_cell]
        # 如果没有特殊指定，这里直接设成全部为0
        init_for_both = tf.matmul(self.incoming[:,0,:], tf.zeros([self.D_input, self.D_cell]))
        self.hid_init = init_for_both
        self.cell_init = init_for_both
        # 所以要将hidden state和memory并在一起。
        self.previous_h_c_tuple = tf.stack([self.hid_init, self.cell_init])
        # 需要将数据由[n_samples, n_steps, D_cell]的形状变成[n_steps, n_samples, D_cell]的形状
        self.incoming = tf.transpose(self.incoming, perm=[1,0,2])

    def one_step(self, previous_h_c_tuple, current_x):
        # 再将hidden state和memory cell拆分开
        prev_h, prev_c = tf.unstack(previous_h_c_tuple)
        # 这时，current_x是当前的输入，
        # prev_h是上一个时刻的hidden state
        # prev_c是上一个时刻的memory cell
        # 计算输入门
        i = tf.sigmoid(
          tf.matmul(current_x, self.W_xi) +
          tf.matmul(prev_h, self.W_hi) +
          self.b_i)
        # 计算遗忘门
        f = tf.sigmoid(
          tf.matmul(current_x, self.W_xf) +
          tf.matmul(prev_h, self.W_hf) +
          self.b_f)
        # 计算输出门
        o = tf.sigmoid(
          tf.matmul(current_x, self.W_xo) +
          tf.matmul(prev_h, self.W_ho) +
          self.b_o)
        # 计算新的数据来源
        c = tf.tanh(
          tf.matmul(current_x, self.W_xc) +
          tf.matmul(prev_h, self.W_hc) +
          self.b_c)
        # 计算当前时刻的memory cell
        current_c = f * prev_c + i * c
        # 计算当前时刻的hidden state
        current_h = o * tf.tanh(current_c)
        # 再次将当前的hidden state和memory cell并在一起返回
        return tf.stack([current_h, current_c])


    def all_steps(self):
        # 输出形状 : [n_steps, n_sample, D_cell]
        h_states = tf.scan(fn=self.one_step,
                          elems=self.incoming,  # 形状为[n_steps, n_sample, D_input]
                          initializer=self.previous_h_c_tuple,
                          name='hstates')[:, 0, :, :]
        return h_states
