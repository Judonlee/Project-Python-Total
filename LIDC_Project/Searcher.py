import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    # loadpath = 'D:/ProjectData/LIDC/Result-Grid/LBP_P=24_R=3/'
    # for C in [1, 10, 100, 1000]:
    #     for gamma in [0.01, 0.001, 0.0001]:
    #         data = numpy.genfromtxt(fname=loadpath + 'C=%d-gamma=%s.csv' % (C, str(gamma)), dtype=float, delimiter=',')[
    #                0:-1]
    #         print(numpy.average(data), end='\t')
    #     print()

    loadpath = 'D:/ProjectData/LIDC/Result-Grid/DX/'
    list = []
    xlabel = []
    for dx in range(5, 101, 5):
        data = numpy.genfromtxt(loadpath + 'DX%04d.csv' % dx, dtype=float, delimiter=',')[0:-1]
        list.append(numpy.average(data))
        xlabel.append(dx)
        for index in range(10):
            print(data[index], end='\t')
        print()
    # print(list)
    # print(dx)
    plt.plot(xlabel, list)
    plt.show()
