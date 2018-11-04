import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:/临时缓存/Records-Result-BLSTM-CTC-CRF-Improve-WA-Train/Bands-120-0/'
    waList, uaList = [], []
    for filename in os.listdir(loadpath):
        if not os.path.isfile(loadpath + filename): continue
        print(filename)
        data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
        # print(data)
        WA, UA = 0, 0
        for index in range(len(data)):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
        WA = WA / sum(sum(data))
        UA = UA / len(data)

        waList.append(WA)
        uaList.append(UA)
    plt.plot(waList, label='Train - WA')
    plt.plot(uaList, label='Train - UA')

    loadpath = 'D:/临时缓存/Bands-120-0/'
    waList, uaList = [], []
    for filename in os.listdir(loadpath):
        if not os.path.isfile(loadpath + filename): continue
        print(filename)
        data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
        # print(data)
        WA, UA = 0, 0
        for index in range(len(data)):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
        WA = WA / sum(sum(data))
        UA = UA / len(data)

        waList.append(WA)
        uaList.append(UA)
    plt.plot(waList, label='Test - WA')
    plt.plot(uaList, label='Test - UA')
    plt.legend()
    plt.show()
    print(waList)
    print(uaList)
