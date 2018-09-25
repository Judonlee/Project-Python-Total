import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'F:\\Project-CTC-Data\\Csv\\Bands60\\'
    savepath = 'F:\\Project-CTC-Data\\Csv-Normalized\\Bands60\\'
    totalData = []
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        print(indexA, indexB, indexC, indexD, indexE)
                        currentData = numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')
                        totalData.extend(currentData)
    print(numpy.shape(totalData))
    totalData = scale(totalData)

    startPosition = 0
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        print(indexA, indexB, indexC, indexD, indexE)
                        currentData = numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')

                        file = open(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                    'w')
                        for indexX in range(len(currentData)):
                            for indexY in range(len(totalData[indexX + startPosition])):
                                if indexY != 0: file.write(',')
                                file.write(str(totalData[indexX + startPosition][indexY]))
                            file.write('\n')
                        file.close()

                        startPosition += len(currentData)
