# from tensorflow.python import pywrap_tensorflow
# import os
#
# checkpoint_path = os.path.join(
#     r'E:\ProjectData_Depression\Experiment\AttentionTransform\RMSE\LA_Both_L1_100\0099-Network')
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print('tensor_name: ', key)

import matplotlib.pylab as plt
import numpy

if __name__ == '__main__':
    print(numpy.sin(numpy.arange(-10, 10, 0.1)))
    plt.subplot(211)
    plt.plot(numpy.sin(numpy.arange(-10, 10, 0.1)))
    plt.axis('off')
    plt.title('Teacher\'s Attention Map', fontsize=30)
    plt.subplot(212)
    plt.plot(numpy.sin(numpy.arange(-10, 10, 1)))
    plt.axis('off')
    plt.title('Stuent\'s Attention Map', fontsize=30)
    plt.show()
