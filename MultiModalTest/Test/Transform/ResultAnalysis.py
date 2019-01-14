import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_SpeechRecognition/Transform/IEMOCAP-Tran-LA-3-Punishment-10000-Result/'
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            if not os.path.exists(os.path.join(loadpath, 'Session%d-%s' % (session, gender))):
                print()
                continue
            uaList, waList = [], []
            for episode in range(100):
                data = numpy.genfromtxt(
                    fname=os.path.join(loadpath, 'Session%d-%s' % (session, gender), 'Decode-%04d.csv' % episode),
                    dtype=int, delimiter=',')

                uaCounter, waCounter = 0, 0
                for index in range(len(data)):
                    uaCounter += data[index][index] / sum(data[index])
                    waCounter += data[index][index]
                uaCounter /= len(data)
                waCounter /= sum(sum(data))
                # print(uaCounter, waCounter)
                uaList.append(uaCounter)
                waList.append(waCounter)
            # plt.plot(uaList, label='UA')
            # plt.plot(waList, label='WA')
            # plt.legend()
            # plt.show()
            print(max(uaList), '\t', max(waList))
