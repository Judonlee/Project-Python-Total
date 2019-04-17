import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    treatpart = 'SA-0-sentence'
    loadpath = 'E:/ProjectData_Depression/SpeechLevel/%s-%s.csv'
    savepath = 'E:/ProjectData_Depression/SpeechLevel-Normalization/%s-%s.csv'

    totalData = []
    totalData.extend(numpy.genfromtxt(fname=loadpath % (treatpart, 'Train'), dtype=float, delimiter=','))
    totalData.extend(numpy.genfromtxt(fname=loadpath % (treatpart, 'Test'), dtype=float, delimiter=','))

    totalData = scale(totalData)

    startPosition = 0
    data = numpy.genfromtxt(fname=loadpath % (treatpart, 'Train'), dtype=float, delimiter=',')
    with open(savepath % (treatpart, 'Train'), 'w') as file:
        for indexX in range(numpy.shape(data)[0]):
            for indexY in range(numpy.shape(data)[1]):
                if indexY != 0: file.write(',')
                file.write(str(totalData[startPosition + indexX][indexY]))
            file.write('\n')
    startPosition += len(data)

    data = numpy.genfromtxt(fname=loadpath % (treatpart, 'Test'), dtype=float, delimiter=',')
    with open(savepath % (treatpart, 'Test'), 'w') as file:
        for indexX in range(numpy.shape(data)[0]):
            for indexY in range(numpy.shape(data)[1]):
                if indexY != 0: file.write(',')
                file.write(str(totalData[startPosition + indexX][indexY]))
            file.write('\n')
