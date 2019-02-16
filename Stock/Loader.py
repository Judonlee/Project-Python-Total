import numpy
from sklearn.preprocessing import scale
import os


def Normalization(loadpath):
    with open(loadpath, 'r') as file:
        data = file.readlines()

    normalizedData, meanList, stdList = [], [], []
    for sample in data:
        result = sample.split('\t')[1:]
        if len(result) == 0: continue
        result[-1] = result[-1][0:-1]

        listData = []
        for sample in result:
            listData.append(float(sample))
        # print(listData)
        normalizedData.append(listData)
    meanList = numpy.mean(normalizedData, axis=0)
    stdList = numpy.std(normalizedData, axis=0)
    normalizedData = scale(normalizedData)
    return normalizedData, meanList, stdList


def Load(partName):
    loadpath = 'E:/ProjectData_Stock/'
    data = numpy.genfromtxt(fname=os.path.join(loadpath, partName + '-Normalization.csv'), dtype=float, delimiter=',')
    print('Load Completed :', numpy.shape(data))
    return data


if __name__ == '__main__':
    normalizedData, meanList, stdList = Normalization(loadpath='E:/ProjectData_Stock/SZ#002415.txt')

    with open('E:/ProjectData_Stock/SZ#002415-Normalization.csv', 'w') as file:
        for indexX in range(numpy.shape(normalizedData)[0]):
            print('Writing %d/%d' % (indexX, numpy.shape(normalizedData)[0]))
            for indexY in range(numpy.shape(normalizedData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(normalizedData[indexX][indexY]))
            file.write('\n')
    with open('E:/ProjectData_Stock/SZ#002415-Parameter.csv', 'w') as file:
        for index in range(len(meanList)):
            if index != 0: file.write(',')
            file.write(str(meanList[index]))
        file.write('\n')
        for index in range(len(stdList)):
            if index != 0: file.write(',')
            file.write(str(stdList[index]))
    # Load('SH#600050')
