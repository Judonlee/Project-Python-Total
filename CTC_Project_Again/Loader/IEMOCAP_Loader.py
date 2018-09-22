import os
import numpy


def IEMOCAP_Loader(loadpath, appoint):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            print('Loading : ', indexA, indexB)
            for indexC in range(1, 6):
                currentData = numpy.load(loadpath + indexA + '\\' + indexB + '\\Session' + str(indexC) + '-Data.npy')
                currentLabel = numpy.load(loadpath + indexA + '\\' + indexB + '\\Session' + str(indexC) + '-Label.npy')

                if ['Female', 'Male'].index(indexB) * 5 + indexC - 1 == appoint:
                    testData.extend(currentData)
                    testLabel.extend(currentLabel)
                else:
                    trainData.extend(currentData)
                    trainLabel.extend(currentLabel)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


if __name__ == '__main__':
    loadpath = 'F:\\Project-CTC-Data\\Csv\\Bands120\\'
    savepath = 'F:\\Project-CTC-Data\\Npy\\Bands120\\'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                currentData, currentLabel = [], []

                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        print(indexA, indexB, indexC, indexD, indexE)
                        treatData = numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')

                        if indexD == 'ang': treatLabel = [1, 0, 0, 0]
                        if indexD == 'exc': treatLabel = [0, 1, 0, 0]
                        if indexD == 'hap': treatLabel = [0, 1, 0, 0]
                        if indexD == 'neu': treatLabel = [0, 0, 1, 0]
                        if indexD == 'sad': treatLabel = [0, 0, 0, 1]
                        # print(numpy.shape(treatData))

                        currentData.append(treatData)
                        currentLabel.append(treatLabel)

                if not os.path.exists(savepath + indexA + '\\' + indexB):
                    os.makedirs(savepath + indexA + '\\' + indexB)
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '-Data.npy', currentData)
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '-Label.npy', currentLabel)
