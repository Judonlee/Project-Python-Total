# -*-coding:utf8-*-

__author = "buyizhiyou"
__date = "2017-11-21"

'''
单步调试,结合汉字的识别学习lstm，ctc loss的tf实现,tensorflow1.4
'''
import tensorflow as tf
import numpy as np
import random


def create_sparse(batch_size, dtype=np.int32):
    '''
    创建稀疏张量,ctc_loss中labels要求是稀疏张量,随机生成序列长度在150～180之间的labels
    '''
    indices = []
    values = []
    for i in range(batch_size):
        length = random.randint(150, 180)
        for j in range(length):
            indices.append((i, j))
            value = random.randint(0, 30)
            values.append(value)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([batch_size, np.asarray(indices).max(0)[1] + 1], dtype=np.int64)  # [64,180]

    return [indices, values, shape]


W = tf.Variable(tf.truncated_normal([200, 781], stddev=0.1),
                name="W")  # num_hidden=200,num_classes=781(想象成780个汉字+blank),shape (200,781)
b = tf.Variable(tf.constant(0., shape=[781]), name="b")  # 781
global_step = tf.Variable(0, trainable=False)  # 全局步骤计数

# 构造输入
inputs = tf.random_normal(shape=[64, 60, 3000], dtype=tf.float32)
# 为了测试，随机batch_size=64张图片,h=60,w=3000,w可以看成lstm的时间步，即lstm输入的time_step=3000,h看成是每一时间步的输入tensor的size
shape = tf.shape(inputs)  # array([ 64, 3000, 60], dtype=int32)
batch_s, max_timesteps = shape[0], shape[1]  # 64,3000
output = create_sparse(64)  # 创建64张图片对应的labels,稀疏张量，序列长度变长
seq_len = np.ones(64) * 180  # 180为变长序列的最大值
labels = tf.SparseTensor(values=output[1], indices=output[0], dense_shape=output[2])

# pdb.set_trace()
cell = tf.nn.rnn_cell.LSTMCell(200, state_is_tuple=True)
inputs = tf.transpose(inputs, [0, 2, 1])
# 转置，因为默认的tf.nn.dynamic_rnn中参数time_major=false,即inputs的shape 是`[batch_size, max_time, ...]`,

'''
tf.nn.dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=None, paralle
l_iterations=None, swap_memory=False, time_major=False, scope=None)
'''
outputs1, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len,
                                dtype=tf.float32)  # (64, 3000, 200)动态rnn实现了输入变长问题的解决方案http://blog.csdn.net/u010223750/article/details/71079036

outputs = tf.reshape(outputs1, [-1, 200])  # (64×3000,200)
logits0 = tf.matmul(outputs, W) + b
logits1 = tf.reshape(logits0, [batch_s, -1, 781])
logits = tf.transpose(logits1, (1, 0, 2))  # (3000, 64, 781)

'''
tf.nn.ctc_loss(labels, inputs, sequence_length, preprocess_collapse_repeated=False, ctc_merge
_repeated=True, ignore_longer_outputs_than_inputs=False, time_major=True)
'''
loss = tf.nn.ctc_loss(labels, logits, seq_len)  # 关于ctc loss解决rnn输出和序列不对齐问题
# http://blog.csdn.net/left_think/article/details/76370453
# https://zhuanlan.zhihu.com/p/23293860
cost = tf.reduce_mean(loss)
optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost, global_step=global_step)
# decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)#or "tf.nn.ctc_greedy_decoder"一种解码策略
# acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(outputs.get_shape())
    print(sess.run(loss))
