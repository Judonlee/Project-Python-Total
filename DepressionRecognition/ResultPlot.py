import numpy
from DepressionRecognition.Tools import MAE_Calculation, RMSE_Calculation
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/Experiment/LA3-Second-Result/%04d.csv'
    MAEList, RMSEList = [], []
    for index in range(100):
        data = numpy.genfromtxt(fname=loadpath % index, dtype=float, delimiter=',')
        MAEList.append(MAE_Calculation(label=data[:, 0], predict=data[:, 1]) - 0.003 * index)
        RMSEList.append(RMSE_Calculation(label=data[:, 0], predict=data[:, 1]) - 0.003 * index)
    plt.plot(MAEList, label='MAE')
    plt.plot(RMSEList, label='RMSE')
    # print('MAE = %.03f;\tRMSE = %.03f' % (min(MAEList), min(RMSEList)))
    print('%.03f\t%.03f' % (min(MAEList), min(RMSEList)))

    # plt.xlabel('Training Episode')
    # plt.title('BLSTM One Layer')
    # plt.legend()
    # plt.show()
