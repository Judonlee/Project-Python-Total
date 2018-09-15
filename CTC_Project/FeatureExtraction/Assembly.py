import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\IEMOCAP-Label-Words\\'
    savepath = 'D:\\ProjectData\\IEMOCAP-Label-Words-Npy\\'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC)
                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    print(indexA, indexB, indexC, indexD)
                    totalData = []

                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        data = numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=int, delimiter=',')
                        totalData.append(len(data))
                    print(totalData)
                    numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '-WordsLabels.npy',
                               totalData)
