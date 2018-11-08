import numpy
import matplotlib.pylab as plt
import os

if __name__ == '__main__':
    loadpath = 'D:/GitHub/CTC_Project_Again/TrainRestart/Tester/Bands-30-Session-1-Female/'
    uaList, waList = [], []

    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
        WA, UA = 0, 0
        for index in range(len(data)):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
        WA = WA / sum(sum(data))
        UA = UA / len(data)
        waList.append(WA)
        uaList.append(UA)
    plt.plot(uaList, label='Train UA')
    plt.plot(waList, label='Train WA')

    loadpath = 'D:/GitHub/CTC_Project_Again/TrainRestart/Tester/Result_Single_BLSTM_CTC_CRF/Bands-30-Session-1-Female/'
    uaList, waList = [], []

    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
        WA, UA = 0, 0
        for index in range(len(data)):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
        WA = WA / sum(sum(data))
        UA = UA / len(data)
        waList.append(WA)
        uaList.append(UA)
    plt.plot(uaList, label='Test UA')
    plt.plot(waList, label='Test WA')
    plt.legend()
    plt.show()
