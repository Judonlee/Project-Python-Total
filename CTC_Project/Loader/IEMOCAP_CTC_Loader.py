import os
import numpy


def IEMOCAP_CTC_Loader(bands, appoint=0):
    datapath = 'D:\\ProjectData\\IEMOCAP-Npy\\Bands' + str(bands) + '\\'
    wordspath = 'D:\\ProjectData\\IEMOCAP-Label-Words-Npy\\'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []

    for indexA in os.listdir(datapath):
        for indexB in os.listdir(datapath + indexA):
            for indexC in os.listdir(datapath + indexA + '\\' + indexB):
                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    data = numpy.load(datapath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '-Data.npy')
                    seq = numpy.load(datapath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '-Seq.npy')
                    label = []

                    words = numpy.load(
                        wordspath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '-WordsLabels.npy')

                    if indexD == 'ang': appointLabel = 1
                    if indexD == 'exc' or indexD == 'hap': appointLabel = 2
                    if indexD == 'neu': appointLabel = 3
                    if indexD == 'sad': appointLabel = 4

                    for sample in words:
                        label.append(numpy.ones(sample, dtype=numpy.int32) * appointLabel)

                    counter = ['Female', 'Male'].index(indexB) * 5 + \
                              ['Session1', 'Session2', 'Session3', 'Session4', 'Session5'].index(indexC)
                    if appoint == counter:
                        testData.extend(data)
                        testSeq.extend(seq)
                        testLabel.extend(label)
                    else:
                        trainData.extend(data)
                        trainSeq.extend(seq)
                        trainLabel.extend(label)
                    print('Loading :', indexA, indexB, indexC, indexD, counter)
    print('Train Part :', numpy.shape(trainData), numpy.shape(trainSeq), numpy.shape(trainLabel))
    print('Test Part :', numpy.shape(testData), numpy.shape(testSeq), numpy.shape(testLabel))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


if __name__ == '__main__':
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = IEMOCAP_CTC_Loader(bands=30)
    print(testLabel)
