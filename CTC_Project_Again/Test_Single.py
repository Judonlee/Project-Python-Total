import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/Visualization/0/1.csv'
    data = numpy.genfromtxt(loadpath, dtype=float, delimiter=',')
    data = numpy.transpose(data, (1, 0))
    print(numpy.shape(data))
    plt.imshow(data, cmap='gray')
    plt.show()
