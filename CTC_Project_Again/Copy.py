import numpy
import matplotlib.pylab as plt
import os

if __name__ == '__main__':
    loadpath = 'D:/Data-Loss-01-Single-BLSTM/Bands-30-Session-2-Female/'
    totalList = []
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(loadpath + filename, dtype=float, delimiter=',')
        totalList.append(data)
    plt.plot(totalList)
    plt.show()
    print(min(totalList))
    print(numpy.argmin(totalList))
