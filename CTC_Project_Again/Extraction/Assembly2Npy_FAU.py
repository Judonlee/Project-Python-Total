import os
import numpy

if __name__ == '__main__':
    bands = 120
    loadpath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class2-Csv-Normalized/Bands-' + str(bands) + '/'
    savepath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class2-Npy/Bands-' + str(bands) + '/'
    os.makedirs(savepath)
    for indexA in os.listdir(loadpath):
        totalData, totalLabel, totalSeq = [], [], []
        for indexB in os.listdir(loadpath + indexA):
            print(bands, indexA, indexB)
            for indexC in os.listdir(loadpath + indexA + '/' + indexB):
                currentData = numpy.genfromtxt(loadpath + indexA + '/' + indexB + '/' + indexC, dtype=float,
                                               delimiter=',')
                totalData.append(currentData)
                totalSeq.append(len(currentData))

                if indexB == 'IDL': currentLabel = [1, 0]
                if indexB == 'NEG': currentLabel = [0, 1]
                if indexB == 'A': currentLabel = [1, 0, 0, 0, 0]
                if indexB == 'E': currentLabel = [0, 1, 0, 0, 0]
                if indexB == 'N': currentLabel = [0, 0, 1, 0, 0]
                if indexB == 'P': currentLabel = [0, 0, 0, 1, 0]
                if indexB == 'R': currentLabel = [0, 0, 0, 0, 1]
                totalLabel.append(currentLabel)
        print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.shape(totalSeq), numpy.sum(totalLabel))
        numpy.save(savepath + indexA + '.npy', [totalData, totalLabel, totalSeq])
