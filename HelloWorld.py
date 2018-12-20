import matplotlib.pylab as plt
import numpy

if __name__ == '__main__':
    loadpath = 'D:/Project-Matlab/Treatment/新建文件夹/%s/Ratio%d.txt'
    for ratio in range(50, 100, 10):
        totalData = numpy.tile([[5]], [30, 91]).tolist()
        for counter in range(30, 301, 1):
            current = []
            if counter % 100 != 0:
                data = numpy.genfromtxt(loadpath % (str(counter / 100), ratio), dtype=float, delimiter=',')
            else:
                data = numpy.genfromtxt(loadpath % (str(int(counter / 100)), ratio), dtype=float, delimiter=',')
            print(counter, numpy.shape(numpy.reshape(data, [-1])))
            current.extend(numpy.reshape(data, [-1]))
            totalData.append(current)
        print(numpy.shape(totalData))
        totalData = numpy.array(totalData)
        totalData = numpy.transpose(totalData, [1, 0])
        print(numpy.average(numpy.reshape(totalData, [-1])))

        # part1 = plt.subplot(330 + ratio / 10)
        plt.imshow(totalData, cmap='gray')
        plt.colorbar()

        plt.xlabel('Slope')
        plt.ylabel('XNob')
        plt.title('Ratio %d' % ratio)
        plt.show()
        exit()
    #plt.colorbar().set_label('Epsilon15NAR:Epsilon15NXR')
    # plt.colorbar()
    # plt.show()

    # plt.ylim([-5, 5])
    # plt.plot(totalData)
    # plt.show()

    # print(totalData[:, 1])
