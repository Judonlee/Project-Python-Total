import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/Experiment/AttentionTransform/MAE/SA_First_L1_100000/%04d.csv'
    totalData = []
    for index in range(100):
        if not os.path.exists(loadpath % (index + 1)): continue
        data = numpy.genfromtxt(fname=loadpath % index, dtype=float, delimiter=',')
        data = numpy.average(data, axis=0)
        totalData.append(data)
        print(data)
    totalData = numpy.array(totalData)
    plt.plot(totalData[:, 0], label='Train Loss')
    plt.plot(totalData[:, 1], label='Attention Transform Loss')
    plt.legend()
    plt.show()
