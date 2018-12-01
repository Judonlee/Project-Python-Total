from sklearn.preprocessing import scale
import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/Voice-Features/Bands-40/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/Voice-Normalized/Bands-40/'

    totalData = []

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                print(indexA, indexB, indexC, numpy.shape(totalData))
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    totalData.extend(
                        numpy.genfromtxt(fname=os.path.join(loadpath, indexA, indexB, indexC, indexD),
                                         dtype=float, delimiter=','))
    print(numpy.shape(totalData))

    totalData = scale(totalData)

    startPosition = 0
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                os.makedirs(os.path.join(savepath, indexA, indexB, indexC))
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    currentData = numpy.genfromtxt(
                        fname=os.path.join(loadpath, indexA, indexB, indexC, indexD),
                        dtype=float, delimiter=',')

                    file = open(os.path.join(savepath, indexA, indexB, indexC, indexD), 'w')
                    for indexX in range(startPosition, startPosition + len(currentData)):
                        for indexY in range(len(totalData[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(totalData[indexX][indexY]))
                        file.write('\n')
                    file.close()

                    startPosition += len(currentData)
