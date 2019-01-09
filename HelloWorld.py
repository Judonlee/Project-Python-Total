import tensorflow as tf
import os
import numpy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 计算Wx_plus_b 的均值与方差,其中axis = [0] 表示想要标准化的维度
img_shape = [128, 32, 32, 64]
Wx_plus_b = tf.Variable(tf.random_normal(img_shape))
axis = list(range(len(img_shape) - 1))
wb_mean, wb_var = tf.nn.moments(Wx_plus_b, axis)

scale = tf.Variable(tf.ones([64]))
offset = tf.Variable(tf.zeros([64]))
variance_epsilon = 0.001
Wx_plus_b = tf.nn.batch_normalization(Wx_plus_b, wb_mean, wb_var, offset, scale, variance_epsilon)

Wx_plus_b1 = (Wx_plus_b - wb_mean) / tf.sqrt(wb_var + variance_epsilon)
Wx_plus_b1 = Wx_plus_b1 * scale + offset

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    print('*** wb_mean ***')
    print(sess.run(wb_mean))
    print('*** wb_var ***')
    print(sess.run(wb_var))
    print('*** Wx_plus_b ***')
    result = sess.run(Wx_plus_b)
    print(numpy.shape(result), numpy.sum(result), numpy.average(result), numpy.std(result, axis=0))
    print('**** Wx_plus_b1 ****')
    result = sess.run(Wx_plus_b)
    print(numpy.shape(result), numpy.sum(result), numpy.average(result), numpy.std(result, axis=0))
