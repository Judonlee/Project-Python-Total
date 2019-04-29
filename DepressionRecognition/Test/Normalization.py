import numpy
import os
from sklearn.preprocessing import scale

if __name__ == '__main__':
    part = 'None-0-sentence'
    loadpath = 'D:/GitHub/DepressionRecognition/Test/%s-%s.csv/%04d.csv'
    savepath = 'E:/ProjectData_Depression/Step5_Assembly/SentenceLevel/%s-%s.npy'

    totalData = []
    for index in range(142):
        data = numpy.genfromtxt(fname=loadpath % (part, 'Train', index), dtype=float, delimiter=',')
        print(index, numpy.shape(data))
        totalData.extend(data)
    for index in range(47):
        data = numpy.genfromtxt(fname=loadpath % (part, 'Test', index), dtype=float, delimiter=',')
        print(index, numpy.shape(data))
        totalData.extend(data)

    print(numpy.shape(totalData))
    totalData = scale(totalData)

    trainData, testData = [], []

    startPosition = 0
    for index in range(142):
        data = numpy.genfromtxt(fname=loadpath % (part, 'Train', index), dtype=float, delimiter=',')
        trainData.append(totalData[startPosition:startPosition + numpy.shape(data)[0]])
        startPosition += numpy.shape(data)[0]
    for index in range(47):
        data = numpy.genfromtxt(fname=loadpath % (part, 'Test', index), dtype=float, delimiter=',')
        testData.append(totalData[startPosition:startPosition + numpy.shape(data)[0]])
        startPosition += numpy.shape(data)[0]
    numpy.save(savepath % (part, 'Train'), trainData)
    numpy.save(savepath % (part, 'Test'), testData)
