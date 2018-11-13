import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:/GitHub/CTC_Project_Again/TrainRestart/Tester/Result-Train/Bands-30-Session-1/%04d.csv'

    waList, uaList = [], []
    for episode in range(90):
        if not os.path.exists(loadpath % (episode)): continue
        data = numpy.genfromtxt(
            fname=loadpath % (episode), dtype=float, delimiter=',')
        WA, UA = 0, 0
        for index in range(len(data)):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
        WA = WA / sum(sum(data))
        UA = UA / len(data)

        waList.append(WA)
        uaList.append(UA)

    plt.plot(waList, label='Train WA')
    plt.plot(uaList, label='Train UA')


    loadpath = 'D:/GitHub/CTC_Project_Again/TrainRestart/Tester/Result/Bands-30-Session-1/%04d.csv'

    waList, uaList = [], []
    for episode in range(90):
        if not os.path.exists(loadpath % (episode)): continue
        data = numpy.genfromtxt(
            fname=loadpath % (episode), dtype=float, delimiter=',')
        WA, UA = 0, 0
        for index in range(len(data)):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
        WA = WA / sum(sum(data))
        UA = UA / len(data)

        waList.append(WA)
        uaList.append(UA)

    plt.plot(waList, label='Test WA')
    plt.plot(uaList, label='Test UA')
    plt.legend()
    plt.show()
    print(numpy.argmax(waList))
