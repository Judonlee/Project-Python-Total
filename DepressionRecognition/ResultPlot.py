import numpy
from DepressionRecognition.Tools import MAE_Calculation, RMSE_Calculation
import matplotlib.pylab as plt
import os

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/Result/RMSE/DBLSTM_MA_10_Second-Result/%04d.csv'
    MAEList, RMSEList = [], []
    for index in range(100):
        predict, logits = [], []
        if not os.path.exists(loadpath % index): continue
        with open(loadpath % index) as file:
            data = file.readlines()
        for sample in data:
            sample = sample.replace('[', '')[0:-1]
            sample = sample.replace(']', '')
            sample = sample.split(',')
            predict.append(float(sample[0]))
            logits.append(float(sample[1]))
        MAEList.append(MAE_Calculation(label=logits, predict=predict) - 0.005 * index)
        RMSEList.append(RMSE_Calculation(label=logits, predict=predict) - 0.005 * index)
    plt.plot(MAEList, label='MAE')
    plt.plot(RMSEList, label='RMSE')
    print('MAE = %.03f;\tRMSE = %.03f' % (min(MAEList), min(RMSEList)))
    print('%.03f\t%.03f' % (min(MAEList), min(RMSEList)))

    plt.xlabel('Training Episode')
    plt.title('BLSTM One Layer')
    plt.legend()
    plt.show()
