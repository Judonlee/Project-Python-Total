import matplotlib.pylab as plt
import numpy

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/Test2/%04d.csv'
    data = []
    for index in range(100):
        currentData = numpy.genfromtxt(loadpath % index, dtype=float, delimiter=',')
        data.extend(currentData)
    plt.plot(data)
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Frame Level Feasibility Experiment')
    plt.show()
