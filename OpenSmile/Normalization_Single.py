import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\Project-CTC-Data\\Csv-Single\\Bands100\\'
    savepath = 'D:\\ProjectData\\Project-CTC-Data\\Csv-Single-Normalized\\Bands100\\'
    totalData = []
    for indexA in ['improve', 'script']:
        for indexB in ['Female', 'Male']:
            for indexC in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
                print(indexA, indexB, indexC)
                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        currentData = numpy.genfromtxt(
                            fname=loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')
                        totalData.append(currentData)
    print(numpy.shape(totalData))
    totalData = scale(totalData)

    startPosition = 0
    for indexA in ['improve', 'script']:
        for indexB in ['Female', 'Male']:
            for indexC in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
                print(indexA, indexB, indexC)
                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        file = open(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                    'w')
                        for indexX in range(len(totalData[startPosition])):
                            if indexX != 0: file.write(',')
                            file.write(str(totalData[startPosition][indexX]))
                        file.close()
                        startPosition += 1
