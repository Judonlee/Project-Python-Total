import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    x = [0, 6, 12, 24, 72]
    y = [6.792275, 12.4599, 15.59706, 18.97337, 20.90421]
    plt.plot(x, y)

    func = numpy.polyfit(x, y, 4)
    print(func)
    formula = numpy.poly1d(func)
    # print(result)
    predict = formula(numpy.arange(0, 72))
    plt.plot(predict)
    plt.show()
