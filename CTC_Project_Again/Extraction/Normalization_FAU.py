import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Csv/Bands-120/'
    savepath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Csv-Normalized/Bands-120/'
    totalData = []
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '/' + indexB):
                currentData = numpy.genfromtxt(loadpath + indexA + '/' + indexB + '/' + indexC, dtype=float,
                                               delimiter=',')
                print(indexA, indexB, indexC, numpy.shape(currentData))
                totalData.extend(currentData)

    print(numpy.shape(totalData))
    totalData = scale(totalData)

    startPosition = 0

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            os.makedirs(savepath + indexA + '/' + indexB)
            for indexC in os.listdir(loadpath + indexA + '/' + indexB):
                currentData = numpy.genfromtxt(loadpath + indexA + '/' + indexB + '/' + indexC, dtype=float,
                                               delimiter=',')
                print('Writing :', indexA, indexB, indexC)
                file = open(savepath + indexA + '/' + indexB + '/' + indexC, 'w')
                for indexX in range(len(currentData)):
                    for indexY in range(len(totalData[startPosition + indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(totalData[startPosition + indexX][indexY]))
                    file.write('\n')
                file.close()
                startPosition += len(currentData)
