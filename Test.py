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
    # name = ['Standard Attention', 'Local Attention(Scope = 1)', 'Local Attention(Scope = 2)', 'Monotonic Attention',
    #         'Monotonic Chunkwise Attention']
    name = ['Without Attention']
    # part = ['SA-0', 'LA-1', 'LA-2', 'MA-10', 'MCA-10']
    part = ['None-0']
    for partIndex in range(len(part)):
        loadpath = 'E:/ProjectData_Depression/Experiment/AutoEncoder-New/%s/' % part[partIndex]
        totalData = []
        for index in range(20):
            data = numpy.genfromtxt(fname=loadpath + '%04d.csv' % index, dtype=float, delimiter=',')
            totalData.extend(data)
        plt.plot(totalData, label=name[partIndex])
    plt.xlabel('Training Batch Number')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
