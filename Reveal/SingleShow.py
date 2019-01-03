import matplotlib.pylab as plt
import matplotlib.colors as colors
import numpy
import math

if __name__ == '__main__':
    drift = 'Epsilon18NXR'
    fold = 'Epsilon15NXR-Epsilon18NXR-Expand'

    maxValue = 15
    minValue = -15

    for ratio in range(10, 100, 10):
        loadpath = 'D:/Project-Matlab/Treatment/新建文件夹/%s-Drifted/%s/%s/Ratio%d.txt'

        totalData = numpy.tile([[-999]], [30, 301]).tolist()
        for counter in range(30, 301, 1):
            current = []
            if counter % 100 != 0:
                data = numpy.genfromtxt(loadpath % (drift, fold, str(counter / 100), ratio), dtype=float, delimiter=',')
            else:
                data = numpy.genfromtxt(loadpath % (drift, fold, str(int(counter / 100)), ratio), dtype=float,
                                        delimiter=',')
            print(counter, numpy.shape(numpy.reshape(data, [-1])))
            current.extend(numpy.reshape(data, [-1]))
            totalData.append(current)
        print(numpy.shape(totalData))
        totalData = numpy.array(totalData)
        totalData = numpy.transpose(totalData, [1, 0])
        print(numpy.average(numpy.reshape(totalData, [-1])))

        totalData = abs(totalData)

        for indexA in range(numpy.shape(totalData)[0]):
            for indexB in range(numpy.shape(totalData)[1]):
                totalData[indexA][indexB] = math.log(totalData[indexA][indexB])

        maxSize = numpy.max(totalData)
        minSize = numpy.min(totalData)
        print(maxSize, minSize)
        for indexA in range(numpy.shape(totalData)[0]):
            for indexB in range(numpy.shape(totalData)[1]):
                if totalData[indexA][indexB] == math.log(999):
                    # totalData[indexA][indexB] = minSize - (maxSize - minSize) / 2
                    totalData[indexA][indexB] = minValue - 5

        print(max(numpy.reshape(totalData, -1)), min(numpy.reshape(totalData, -1)))
        totalData[0][0] = 15
        ####################################################################
        # maxResult = max(numpy.reshape(totalData, -1))
        # totalData = numpy.tile([[maxResult]], [numpy.shape(totalData)[0], numpy.shape(totalData)[1]]) - totalData
        ###########################################################

        ax = plt.subplot(1, 1, 1)
        plt.imshow(totalData, cmap='Blues', origin='lower')
        colorbar = plt.colorbar()
        if fold == 'Epsilon15NAR-Epsilon15NXR':
            colorbar.set_label('$^{15}$ε$_{NAR}$ / $^{15}$ε$_{NXR}$')
        if fold == 'Epsilon15NAR-Epsilon18NXR':
            colorbar.set_label('$^{15}$ε$_{NAR}$ / $^{18}$ε$_{NXR}$')
        if fold == 'Epsilon15NXR-Epsilon18NXR':
            colorbar.set_label('$^{15}$ε$_{NXR}$ / $^{18}$ε$_{NXR}$')

        # colorbar.set_clim(vmin=minSize, vmax=numpy.max(totalData))
        # colorbar.set_array()
        # colorbar.set_ticks([minSize, (minSize + numpy.max(totalData)) / 2, numpy.max(totalData)])
        # norm = colors.Normalize(vmin=minSize, vmax=numpy.max(totalData))
        # colorbar.set_clim(vmin=minSize,vmax=numpy.max(totalData))
        # colorbar.set_ticks(numpy.linspace(minSize, numpy.max(totalData), 3))
        # colorbar.set_ticklabels(
        #     ['%.2f' % minSize, '%.2f' % ((minSize + numpy.max(totalData)) / 2),
        #      '%.2f' % numpy.max(totalData)])
        # position

        colorbar.set_clim(vmin=-15, vmax=15)
        colorbar.set_ticks(numpy.linspace(-15, 15, 11))

        plt.xlabel('Slope')
        plt.ylabel('Exchange ratio')
        plt.title('Ratio %d' % ratio)

        plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
        plt.yticks([0, 75, 150, 225, 300], ['0.00', '0.25', '0.50', '0.75', '1.00'])
        plt.savefig('Ratio-%d.png' % ratio)
        plt.show()
        # exit()
