import tensorflow
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=0.5)
dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=None)
calc = tensorflow.sqrt(x=dataInput)

sess = tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=gpu_options))
sess.run(tensorflow.global_variables_initializer())
while True:
    pass
