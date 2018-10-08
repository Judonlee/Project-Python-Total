import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Csv-Normalized/Bands-120/'
    savepath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Npy/Bands-120/'
    os.makedirs(savepath)
    for indexA in os.listdir(loadpath):
        totalData, totalLabel = [], []
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '/' + indexB):
                print(indexA, indexB, indexC)
                currentData = numpy.genfromtxt(loadpath + indexA + '/' + indexB + '/' + indexC, dtype=float,
                                               delimiter=',')
                totalData.append(currentData)

                # if indexB == 'IDL': currentLabel = [1, 0]
                # if indexB == 'NEG': currentLabel = [0, 1]
                if indexB == 'A': currentLabel = [1, 0, 0, 0, 0]
                if indexB == 'E': currentLabel = [0, 1, 0, 0, 0]
                if indexB == 'N': currentLabel = [0, 0, 1, 0, 0]
                if indexB == 'P': currentLabel = [0, 0, 0, 1, 0]
                if indexB == 'R': currentLabel = [0, 0, 0, 0, 1]
                totalLabel.append(currentLabel)
        print(numpy.shape(totalData), numpy.shape(totalLabel))
        numpy.save(savepath + indexA + '.npy', [totalData, totalLabel])
