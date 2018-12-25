import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/OpenSmile/emobase/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/OpenSmile/emobase-Normalization/'
    totalData = []

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    data = numpy.genfromtxt(fname=os.path.join(loadpath, indexA, indexB, indexC, indexD), dtype=float,
                                            delimiter=',')
                    print(indexA, indexB, indexC, indexD, numpy.shape(data))
                    totalData.append(data)
    print(numpy.shape(totalData))

    totalData = scale(totalData)
    startPosition = 0
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                os.makedirs(os.path.join(savepath, indexA, indexB, indexC))
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    print(indexA, indexB, indexC, indexD)
                    with open(os.path.join(savepath, indexA, indexB, indexC, indexD), 'w') as file:
                        for index in range(numpy.shape(totalData)[1]):
                            if index != 0: file.write(',')
                            file.write(str(totalData[startPosition][index]))
                    startPosition += 1
