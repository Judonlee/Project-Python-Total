import numpy
from DepressionRecognition.Tools import MAE_Calculation, RMSE_Calculation
import matplotlib.pylab as plt
import os

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/NoneAttention/DBLSTM_HA_From_None_To_None_0_sentence_Result/%04d.csv'
    MAEList, RMSEList = [], []
    for index in range(100):
        if not os.path.exists(loadpath % index): continue
        data = numpy.genfromtxt(fname=loadpath % index, dtype=float, delimiter=',')
        MAEList.append(MAE_Calculation(label=data[:, 0], predict=data[:, 1]))
        RMSEList.append(RMSE_Calculation(label=data[:, 0], predict=data[:, 1]))

    # print(MAEList)
    plt.plot(MAEList, label='MAE')
    plt.plot(RMSEList, label='RMSE')
    print('MAE = %.03f;\tRMSE = %.03f' % (min(MAEList), min(RMSEList)))
    print('%.03f\t%.03f' % (min(MAEList), min(RMSEList)))

    # plt.xlabel('Training Episode')
    # plt.title('BLSTM One Layer')
    # plt.legend()
    # plt.show()
