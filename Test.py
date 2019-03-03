import matplotlib.pylab as plt
import numpy

if __name__ == '__main__':
    data = numpy.sin(numpy.arange(-3 * numpy.pi, 3 * numpy.pi, 0.01 * numpy.pi)) + numpy.cos(
        numpy.arange(-30 * numpy.pi, 30 * numpy.pi, 0.1 * numpy.pi))
    plt.plot(data)
    plt.axis('off')
    plt.show()
