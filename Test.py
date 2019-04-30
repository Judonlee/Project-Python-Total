import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    data = numpy.genfromtxt('Result-Whole.csv', dtype=float, delimiter=',')
    plt.plot(data)
    plt.show()
