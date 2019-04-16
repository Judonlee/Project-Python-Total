import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder-Parameter/'
    for foldname in os.listdir(loadpath):
        totalData = []
        for index in range(100):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, foldname, '%04d.csv' % index), dtype=float,
                                    delimiter=',')
            totalData.append(numpy.average(data))
        plt.plot(totalData, label=foldname)
    plt.legend()
    plt.xlabel('Train Episode')
    plt.ylabel('Loss')
    plt.show()
