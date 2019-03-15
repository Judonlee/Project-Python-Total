import matplotlib.pylab as plt
import matplotlib.colors as colors
import numpy
import math

if __name__ == '__main__':
    drift = 'Epsilon18NXR'
    fold = 'Epsilon15NXR-Epsilon18NXR-Expand'

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
            # print(counter, numpy.shape(numpy.reshape(data, [-1])))
            current.extend(numpy.reshape(data, [-1]))
            totalData.append(current)

        totalData = numpy.array(totalData)
        totalData = numpy.transpose(totalData, [1, 0])

        # plt.imshow(totalData)
        # plt.show()
        # exit()

        totalData = abs(totalData)
        print(numpy.shape(totalData))

        validList = []
        for indexA in range(numpy.shape(totalData)[0]):
            for indexB in range(numpy.shape(totalData)[1]):
                if totalData[indexA][indexB] != 999: validList.append(totalData[indexA][indexB])

        # print(max(validList), min(validList))
        if len(validList) == 0: continue
        maxValue = numpy.log(max(validList))
        minValue = numpy.log(min(validList))
        print(maxValue, minValue)

        for indexA in range(numpy.shape(totalData)[0]):
            for indexB in range(numpy.shape(totalData)[1]):
                print(indexA, indexB)
                if totalData[indexA][indexB] == 999.0:
                    totalData[indexA][indexB] = minValue - (maxValue - minValue) / 2
                    continue
                totalData[indexA][indexB] = math.log(totalData[indexA][indexB])

        plt.imshow(totalData, cmap='gray_r', origin='lower')
        # plt.show()

        colorbar = plt.colorbar()
        print(minValue - (maxValue - minValue) / 2, minValue, minValue + (minValue - maxValue) / 2, maxValue)
        colorbar.set_clim(vmin=minValue - (maxValue - minValue) / 2, vmax=maxValue)
        colorbar.set_ticks(
            [minValue - (maxValue - minValue) / 2, minValue, minValue + (maxValue - minValue) / 2, maxValue])
        colorbar.set_ticklabels(
            ['', '%.2f' % minValue, '%.2f' % (minValue + (maxValue - minValue) / 2), '%.2f' % maxValue])

        if fold == 'Epsilon15NAR-Epsilon15NXR-Expand':
            colorbar.set_label('$^{15}$ε$_{NAR}$ / $^{15}$ε$_{NXR}$')
        if fold == 'Epsilon15NAR-Epsilon18NXR-Expand':
            colorbar.set_label('$^{15}$ε$_{NAR}$ / $^{18}$ε$_{NXR}$')
        if fold == 'Epsilon15NXR-Epsilon18NXR-Expand':
            colorbar.set_label('$^{15}$ε$_{NXR}$ / $^{18}$ε$_{NXR}$')

        plt.xlabel('Slope')
        plt.ylabel('Exchange ratio')
        plt.title('Ratio %d' % ratio)

        plt.xticks([0, 50, 100, 150, 200, 250, 300], ['0.0', '0.5', '1.0', '1.5', '2.0', '2.5', '3.0'])
        plt.yticks([0, 75, 150, 225, 300], ['0.00', '0.25', '0.50', '0.75', '1.00'])
        plt.savefig('Ratio-%d.png' % ratio)

        plt.show()

        with open('Ratio-%d.csv' % ratio, 'w') as file:
            for indexA in range(numpy.shape(totalData)[0]):
                for indexB in range(numpy.shape(totalData)[1]):
                    if indexB != 0: file.write(',')
                    if totalData[indexA][indexB] == minValue - (maxValue - minValue) / 2:
                        file.write('ERROR')
                    else:
                        file.write(str(totalData[indexA][indexB]))
                file.write('\n')

        # exit()
