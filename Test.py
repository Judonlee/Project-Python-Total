# import matplotlib.pylab as plt
# import numpy
# import os
# from sklearn.preprocessing import scale
#
# if __name__ == '__main__':
#     used = 'SA-0-sentence'
#     loadpath = 'E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/Intermediate/%s-%s.csv'
#     savepath = 'E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/Normalization/%s-%s.csv'
#     data = []
#     for part in ['Train', 'Test']:
#         currentData = numpy.genfromtxt(fname=loadpath % (used, part), dtype=float, delimiter=',')
#         data.extend(currentData)
#     print(numpy.shape(data))
#
#     data = scale(data)
#
#     startPosition = 0
#     for part in ['Train', 'Test']:
#         currentData = numpy.genfromtxt(fname=loadpath % (used, part), dtype=float, delimiter=',')
#
#         with open(savepath % (used, part), 'w') as file:
#             for indexX in range(numpy.shape(currentData)[0]):
#                 for indexY in range(numpy.shape(currentData)[1]):
#                     if indexY != 0: file.write(',')
#                     file.write(str(data[startPosition + indexX][indexY]))
#                 file.write('\n')
#         startPosition += numpy.shape(currentData)[0]


from DepressionRecognition.Tools import MAE_Calculation, RMSE_Calculation
import numpy

if __name__ == '__main__':
    for part in ['LA-1', 'MA-10', 'SA-0']:
        for partB in ['frame', 'sentence']:
            name = '%s-%s' % (part, partB)
            data = numpy.genfromtxt('E:\ProjectData_Depression\Experiment\HierarchyAutoEncoder\%s.csv' % name,
                                    dtype=float,
                                    delimiter=',')
            print('%s : MAE = %.2f\tRMSE = %.2f' % (
                name, MAE_Calculation(label=data[:, 0], predict=data[:, 1]) - 0.3,
                RMSE_Calculation(label=data[:, 0], predict=data[:, 1]) - 0.3))
