import numpy
import os
from sklearn.preprocessing import scale

if __name__ == '__main__':
    usedpart = 'LA-1-frame'
    loadpath = 'E:/ProjectData_Depression/Experiment/SentenceLevel/%s-%s.csv/'
    os.makedirs(loadpath % (usedpart + '-Normalization', 'Train'))
    os.makedirs(loadpath % (usedpart + '-Normalization', 'Test'))
    totalData = []
    for index in range(142):
        print('Train', index)
        data = numpy.genfromtxt(fname=os.path.join(loadpath % (usedpart, 'Train'), '%04d.csv' % index), dtype=float,
                                delimiter=',')
        totalData.extend(data)
    for index in range(47):
        print('Test', index)
        data = numpy.genfromtxt(fname=os.path.join(loadpath % (usedpart, 'Test'), '%04d.csv' % index), dtype=float,
                                delimiter=',')
        totalData.extend(data)
    print(numpy.shape(totalData))

    totalData = scale(totalData)

    startPosition = 0
    for index in range(142):
        print('Writing Train', index)
        data = numpy.genfromtxt(fname=os.path.join(loadpath % (usedpart, 'Train'), '%04d.csv' % index), dtype=float,
                                delimiter=',')
        with open(loadpath % (usedpart + '-Normalization', 'Train') + '%04d.csv' % index, 'w') as file:
            for indexX in range(numpy.shape(data)[0]):
                for indexY in range(numpy.shape(data)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(totalData[startPosition + indexX][indexY]))
                file.write('\n')
            startPosition += len(data)

    for index in range(47):
        print('Writing Test', index)
        data = numpy.genfromtxt(fname=os.path.join(loadpath % (usedpart, 'Test'), '%04d.csv' % index), dtype=float,
                                delimiter=',')
        with open(loadpath % (usedpart + '-Normalization', 'Test') + '%04d.csv' % index, 'w') as file:
            for indexX in range(numpy.shape(data)[0]):
                for indexY in range(numpy.shape(data)[1]):
                    if indexY != 0: file.write(',')
                    file.write(str(totalData[startPosition + indexX][indexY]))
                file.write('\n')
            startPosition += len(data)
