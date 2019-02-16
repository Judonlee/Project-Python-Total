import numpy
import os


def Load_EncoderDecoder():
    loadpath = 'D:/ProjectData/AVEC2017-Bands40/Step5_2_PartAssembly/'
    trainData = numpy.load(os.path.join(loadpath, 'Train', 'Data.npy'))
    trainLabel = numpy.load(os.path.join(loadpath, 'Train', 'Label.npy'))
    testData = numpy.load(os.path.join(loadpath, 'Develop', 'Data.npy'))
    testLabel = numpy.load(os.path.join(loadpath, 'Develop', 'Label.npy'))

    trainSeq, testSeq, trainLabelSeq, testLabelSeq = [], [], [], []
    for index in range(len(trainData)):
        trainSeq.append(len(trainData[index]))
    for index in range(len(testData)):
        testSeq.append(len(testData[index]))
    for index in range(len(trainLabel)):
        trainLabelSeq.append(len(trainLabel[index]))
    for index in range(len(testLabel)):
        testLabelSeq.append(len(testLabel[index]))

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainLabelSeq),
          numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.shape(testLabelSeq))
    return trainData, trainLabel, trainSeq, trainLabelSeq, testData, testLabel, testSeq, testLabelSeq


def Load_DBLSTM(maxNumber=128, maxLen=1000):
    loadpath = 'D:/ProjectData/AVEC2017-Bands40/Step6_TD_Part/'
    # loadpath = '/mnt/external/Bobs/AttentionTransform/%s/Step6_TD_Part' % conf

    trainLabelData = numpy.genfromtxt(fname='D:/ProjectData/AVEC2017-Bands40/TrainLabel.csv', dtype=int, delimiter=',')
    testLabelData = numpy.genfromtxt(fname='D:/ProjectData/AVEC2017-Bands40/DevelopLabel.csv', dtype=int, delimiter=',')

    # trainLabelData = numpy.genfromtxt(fname='/mnt/external/Bobs/AttentionTransform/TrainLabel.csv', dtype=int,
    #                                   delimiter=',')
    # testLabelData = numpy.genfromtxt(fname='/mnt/external/Bobs/AttentionTransform/DevelopLabel.csv', dtype=int,
    #                                  delimiter=',')

    trainData, trainSeq, trainLabel, testData, testSeq, testLabel = [], [], [], [], [], []
    for indexA in range(len(trainLabelData)):
        data = numpy.load(os.path.join(loadpath, 'Train', '%d.npy' % trainLabelData[indexA][0]))

        batchData, batchSeq, finalBatchSeq = [], [], []
        for sample in data: batchSeq.append(min(len(sample), maxLen))

        for indexB in range(len(data)):
            if numpy.shape(data[indexB])[0] == 0: continue
            if len(finalBatchSeq) >= maxNumber: break

            finalBatchSeq.append(batchSeq[indexB])
            if numpy.shape(data[indexB])[0] > maxLen:
                batchData.append(data[indexB][0:maxLen])
            else:
                batchData.append(numpy.concatenate(
                    [data[indexB],
                     numpy.zeros([max(batchSeq) - numpy.shape(data[indexB])[0], numpy.shape(data[indexB])[1]])]))

        trainData.append(batchData)
        trainSeq.append(finalBatchSeq)
        trainLabel.append(trainLabelData[indexA][2])
        print('Loading Train', trainLabelData[indexA][0], numpy.shape(batchData))

    for indexA in range(len(testLabelData)):
        data = numpy.load(os.path.join(loadpath, 'Develop', '%d.npy' % testLabelData[indexA][0]))

        batchData, batchSeq, finalBatchSeq = [], [], []
        for sample in data: batchSeq.append(len(sample))

        for indexB in range(len(data)):
            if numpy.shape(data[indexB])[0] == 0: continue
            batchData.append(numpy.concatenate(
                [data[indexB],
                 numpy.zeros([max(batchSeq) - numpy.shape(data[indexB])[0], numpy.shape(data[indexB])[1]])]))
            finalBatchSeq.append(batchSeq[indexB])

        testData.append(batchData)
        testSeq.append(finalBatchSeq)
        testLabel.append(testLabelData[indexA][2])
        print('Loading Develop', testLabelData[indexA][0], numpy.shape(batchData))

    print(numpy.shape(trainData), numpy.shape(trainSeq), numpy.shape(trainLabel), numpy.shape(testData),
          numpy.shape(testSeq), numpy.shape(testLabel))
    return trainData, trainSeq, trainLabel, testData, testSeq, testLabel
