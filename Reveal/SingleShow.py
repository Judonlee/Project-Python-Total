import matplotlib.pylab as plt
import numpy
import math

if __name__ == '__main__':
    loadpath = 'D:/Project-Matlab/Treatment/新建文件夹/Epsilon15NXR-Drifted/Epsilon15NAR-Epsilon15NXR/%s/Ratio%d.txt'
    ratio = 10
    totalData = numpy.tile([[-999]], [30, 100]).tolist()
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

    for indexA in range(numpy.shape(totalData)[0]):
        for indexB in range(numpy.shape(totalData)[1]):
            if totalData[indexA][indexB] == -999:
                totalData[indexA][indexB] = 0
    minSize = numpy.min(totalData)
    for indexA in range(numpy.shape(totalData)[0]):
        for indexB in range(numpy.shape(totalData)[1]):
            if totalData[indexA][indexB] == 0:
                totalData[indexA][indexB] = minSize

    # part1 = plt.subplot(330 + ratio / 10)
    totalData = abs(totalData)
    for indexA in range(numpy.shape(totalData)[0]):
        for indexB in range(numpy.shape(totalData)[1]):
            print(totalData[indexA][indexB])
            totalData[indexA][indexB] = math.log(totalData[indexA][indexB])

    print(max(numpy.reshape(totalData, -1)), min(numpy.reshape(totalData, -1)))

    ####################################################################
    # maxResult = max(numpy.reshape(totalData, -1))
    # totalData = numpy.tile([[maxResult]], [numpy.shape(totalData)[0], numpy.shape(totalData)[1]]) - totalData
    ###########################################################

    ax = plt.subplot(1, 1, 1)
    plt.imshow(totalData, cmap='gist_yarg', origin='lower')
    plt.colorbar().set_label('$ε_1$$_5$$_N$$_A$$_R$ / $ε_1$$_5$$_N$$_X$$_R$')

    plt.xlabel('Slope')
    plt.ylabel('Exchange Ratio')
    plt.title('$ε_1$$_5$$_N$$_X$$_R$ Drifted - Ratio %d' % ratio)

    plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
    plt.yticks([0, 25, 50, 75, 100], ['0.00', '0.25', '0.50', '0.75', '1.00'])
    plt.savefig('Ratio-%d.png' % ratio)
    plt.show()
    # exit()
