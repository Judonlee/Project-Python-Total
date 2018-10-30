import os
import numpy

if __name__ == '__main__':
    for appoint in range(10):
        loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Features/IS09-Normalized/'
        savepath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Features/IS09-Npy/Appoint-%d/' % appoint

        os.makedirs(savepath)
        trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []
        for indexA in os.listdir(loadpath):
            for indexB in os.listdir(loadpath + indexA):
                for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
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

                            if ['Female', 'Male'].index(indexB) * 5 + ['Session1', 'Session2', 'Session3', 'Session4',
                                                                       'Session5'].index(indexC) == appoint:
                                testData.append(treatData)
                                testLabel.append(treatLabel)
                                testSeq.append(len(treatData))
                            else:
                                trainData.append(treatData)
                                trainLabel.append(treatLabel)
                                trainSeq.append(len(treatData))
        print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.sum(trainLabel, axis=0))
        print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.sum(testLabel, axis=0))
        numpy.save(savepath + 'TrainData.npy', [trainData, trainLabel, trainSeq, numpy.ones(len(trainData))])
        numpy.save(savepath + 'TestData.npy', [testData, testLabel, testSeq, numpy.ones(len(testData))])
