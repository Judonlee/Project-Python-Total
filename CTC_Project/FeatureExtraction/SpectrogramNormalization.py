from sklearn.preprocessing import scale
import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\IEMOCAP\\IEMOCAP-Seq-Features\\IS10\\'
    savepath = 'D:\\ProjectData\\IEMOCAP\\IEMOCAP-Seq-Features\\IS10-Normalized\\'

    totalData = []

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                for indexD in ['ang', 'exc', 'sad', 'neu', 'hap']:
                    print(indexA, indexB, indexC, indexD, numpy.shape(totalData))
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        totalData.extend(numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=','))
    print(numpy.shape(totalData))

    totalData = scale(totalData)

    startPosition = 0
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                for indexD in ['ang', 'exc', 'sad', 'neu', 'hap']:
                    os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        currentData = numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')

                        file = open(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                    'w')
                        for indexX in range(startPosition, startPosition + len(currentData)):
                            for indexY in range(len(totalData[indexX])):
                                if indexY != 0: file.write(',')
                                file.write(str(totalData[indexX][indexY]))
                            file.write('\n')
                        file.close()

                        startPosition += len(currentData)
