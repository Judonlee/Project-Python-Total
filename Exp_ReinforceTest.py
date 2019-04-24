import tensorflow
import os
import numpy

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.8)
dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[128, 32, 32, 1])
calc = tensorflow.layers.conv2d(inputs=dataInput, filters=1, kernel_size=256, strides=1, padding='SAME', name='calc')

sess = tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))
sess.run(tensorflow.global_variables_initializer())
while True:
    result = sess.run(fetches=calc, feed_dict={dataInput: numpy.ones([128, 32, 32, 1])})
