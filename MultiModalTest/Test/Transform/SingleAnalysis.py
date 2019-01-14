import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_SpeechRecognition/Transform/IEMOCAP-Tran-LA-3-Punishment-1-Result/Session1-Female/'
    totalData = []
    for episode in range(100):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, 'TestLoss-%04d.csv' % episode), dtype=float, delimiter=',')
        totalData.append(data)
    totalData = numpy.array(totalData)
    plt.plot(totalData[:, 2], label='TotalLoss')
    plt.legend()
    plt.show()
