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
import matplotlib.pylab as plt
import os
import librosa

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/Experiment/DBLSTM-AutoEncoder/SA-0/'

    totalList = []
    for index in range(100):
        batchData = numpy.genfromtxt(os.path.join(loadpath, '%04d.csv' % index), dtype=float, delimiter=',')
        totalList.append(numpy.average(batchData))
    plt.plot(totalList)
    plt.xlabel('Training Episode')
    plt.ylabel('Loss')
    plt.show()
