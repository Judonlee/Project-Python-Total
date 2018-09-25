import os
import numpy

if __name__ == '__main__':
    loadpath = 'F:\\Project-CTC-Data\\Csv-Normalized\\Bands30\\'
    savepath = 'F:\\Project-CTC-Data\\Npy-Normalized\\Bands30\\'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            os.makedirs(savepath + indexA + '\\' + indexB)
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                totalData, totalLabel, totalSeq = [], [], []

                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    print(indexA, indexB, indexC, indexD)
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        treatData = numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')
                        treatLabel = numpy.zeros(4)
                        if indexD == 'ang': treatLabel[0] = 1
                        if indexD == 'exc' or indexD == 'hap': treatLabel[1] = 1
                        if indexD == 'neu': treatLabel[2] = 1
                        if indexD == 'sad': treatLabel[3] = 1
                        totalData.append(treatData)
                        totalLabel.append(treatLabel)
                        totalSeq.append(len(treatData))
                print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.sum(totalLabel, axis=0),
                      numpy.shape(totalSeq))
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '-Data.npy', totalData)
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '-Label.npy', totalLabel)
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '-Seq.npy', totalSeq)
