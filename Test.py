# from tensorflow.python import pywrap_tensorflow
# import os
#
# checkpoint_path = os.path.join(
#     r'E:\ProjectData_Depression\Experiment\AttentionTransform\RMSE\LA_Both_L1_100\0099-Network')
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print('tensor_name: ', key)

import numpy

if __name__ == '__main__':
    data = numpy.load(r'D:\LIDC\LBP-Npy\R=1_P=4\Part0-Label.npy')
    print(data)
    print(numpy.shape(data))
