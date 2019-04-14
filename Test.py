import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/Experiment/FrameLevel/DBLSTM_LA-1-frame/'
    totalData = []
    for index in range(46):
        data = numpy.genfromtxt(loadpath + '%04d.csv' % index, dtype=float, delimiter=',')
        totalData.append(numpy.average(data))
    plt.plot(totalData)
    plt.xlabel('Train Episode')
    plt.ylabel('Loss')
    plt.show()
