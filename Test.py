# from tensorflow.python import pywrap_tensorflow
# import os
#
# checkpoint_path = os.path.join(
#     r'E:\ProjectData_Depression\Experiment\AttentionTransform\RMSE\LA_Both_L1_100\0099-Network')
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print('tensor_name: ', key)

import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/Experiment/EncoderDecoder/'
    for foldname in os.listdir(loadpath):
        totalData = []
        for filename in os.listdir(os.path.join(loadpath, foldname)):
            if filename[-3:] != 'csv': continue
            print(foldname, filename)
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, filename), dtype=float, delimiter=',')
            totalData.append(numpy.average(data))
        plt.plot(totalData, label=foldname)
    plt.legend()
    plt.title('Train Loss Line')
    plt.ylabel('Loss')
    plt.xlabel('Training Episode')
    plt.show()
