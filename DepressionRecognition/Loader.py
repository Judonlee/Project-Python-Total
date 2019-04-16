import numpy
import os


def Load_EncoderDecoder():
    loadpath = 'E:/ProjectData_Depression/Step5_Assembly/'
    trainData, trainLabel, trainDataSeq, trainLabelSeq, testData, testLabel, testDataSeq, testLabelSeq = [], [], [], [], [], [], [], []

    for indexA in ['Train', 'Develop', 'Test']:
        for indexB in os.listdir(os.path.join(loadpath, indexA))[0:2]:
            if indexB.find('Data') == -1: continue
            print(indexA, indexB)

            batchData, batchLabel, batchDataSeq, batchLabelSeq = [], [], [], []
            currentData = numpy.load(file=os.path.join(loadpath, indexA, indexB))
            currentLabel = numpy.load(file=os.path.join(loadpath, indexA, indexB.replace('Data', 'Label')))

            for index in range(numpy.shape(currentData)[0]):
                if len(currentLabel[index]) == 0: continue
                batchData.append(currentData[index])
                batchDataSeq.append(numpy.shape(currentData[index])[0])
                batchLabel.append(currentLabel[index])
                batchLabelSeq.append(numpy.shape(currentLabel[index])[0])

            if indexA in ['Train', 'Develop']:
                trainData.extend(batchData)
                trainLabel.extend(batchLabel)
                trainDataSeq.extend(batchDataSeq)
                trainLabelSeq.extend(batchLabelSeq)
            else:
                testData.extend(batchData)
                testLabel.extend(batchLabel)
                testDataSeq.extend(batchDataSeq)
                testLabelSeq.extend(batchLabelSeq)

    print('Train Load Completed', numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainDataSeq),
          numpy.shape(trainLabelSeq))
    print('Test Load Completed', numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testDataSeq),
          numpy.shape(testLabelSeq))
    return trainData, trainLabel, trainDataSeq, trainLabelSeq, testData, testLabel, testDataSeq, testLabelSeq


def Load_DBLSTM(maxSentence=128, maxLen=1000):
    loadpath = 'E:/ProjectData_Depression/Step5_Assembly/'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []

    for indexA in ['Train', 'Develop', 'Test']:
        labelData = numpy.genfromtxt(fname=os.path.join(loadpath, '%sLabel.csv' % indexA), dtype=int, delimiter=',')

        for indexB in range(numpy.shape(labelData)[0]):
            batchData, batchLabel, batchSeq = [], labelData[indexB][2], []

            currentData = numpy.load(file=os.path.join(loadpath, indexA, '%d_P_Data.npy' % labelData[indexB][0]))[
                          0:maxSentence]
            for indexC in range(numpy.shape(currentData)[0]):
                batchSeq.append(min(maxLen, numpy.shape(currentData[indexC])[0]))
            for indexC in range(numpy.shape(currentData)[0]):
                if numpy.shape(currentData[indexC])[0] > maxLen:
                    partData = currentData[indexC][0:maxLen]
                else:
                    partData = numpy.concatenate([currentData[indexC], numpy.zeros(
                        shape=[max(batchSeq) - numpy.shape(currentData[indexC])[0],
                               numpy.shape(currentData[indexC])[1]])], axis=0)
                # print(numpy.shape(partData))
                batchData.append(partData)
            print(indexA, labelData[indexB][0], numpy.shape(batchData), batchLabel, numpy.shape(batchSeq))
            if indexA in ['Train', 'Develop']:
                trainData.append(batchData)
                trainLabel.append([batchLabel])
                trainSeq.append(batchSeq)
            else:
                testData.append(batchData)
                testLabel.append([batchLabel])
                testSeq.append(batchSeq)

    print('Train Load Completed', numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    print('Test Load Completed', numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


def Load_DBLSTM_ChoosePart(part, maxSentence=128, maxLen=1000):
    loadpath = 'E:/ProjectData_Depression/Step5_Assembly/'
    trainData, trainLabel, trainSeq = [], [], []

    for indexA in part:
        labelData = numpy.genfromtxt(fname=os.path.join(loadpath, '%sLabel.csv' % indexA), dtype=int, delimiter=',')

        for indexB in range(numpy.shape(labelData)[0]):
            batchData, batchLabel, batchSeq = [], labelData[indexB][2], []

            currentData = numpy.load(file=os.path.join(loadpath, indexA, '%d_P_Data.npy' % labelData[indexB][0]))[
                          0:maxSentence]
            for indexC in range(numpy.shape(currentData)[0]):
                batchSeq.append(min(maxLen, numpy.shape(currentData[indexC])[0]))
            for indexC in range(numpy.shape(currentData)[0]):
                if numpy.shape(currentData[indexC])[0] > maxLen:
                    partData = currentData[indexC][0:maxLen]
                else:
                    partData = numpy.concatenate([currentData[indexC], numpy.zeros(
                        shape=[max(batchSeq) - numpy.shape(currentData[indexC])[0],
                               numpy.shape(currentData[indexC])[1]])], axis=0)
                # print(numpy.shape(partData))
                batchData.append(partData)
            print(indexA, labelData[indexB][0], numpy.shape(batchData), batchLabel, numpy.shape(batchSeq))
            trainData.append(batchData)
            trainLabel.append([batchLabel])
            trainSeq.append(batchSeq)

    print('Train Load Completed', numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
    return trainData, trainLabel, trainSeq


def Loader_AutoEncoder():
    loadpath = 'E:/ProjectData_Depression/Step5_Assembly/'
    data, seq = [], []

    for indexA in ['Train', 'Develop', 'Test']:
        labelData = numpy.genfromtxt(fname=os.path.join(loadpath, '%sLabel.csv' % indexA), dtype=int, delimiter=',')[
                    0:2]

        for indexB in range(numpy.shape(labelData)[0]):
            currentData = numpy.load(file=os.path.join(loadpath, indexA, '%d_P_Data.npy' % labelData[indexB][0]))
            data.extend(currentData)
            for sample in currentData:
                seq.append(numpy.shape(sample)[0])
    return data, seq


def Loader_SentenceLevel(part):
    loadpath = 'E:/ProjectData_Depression/Step5_Assembly/SentenceLevel/'
    trainData = numpy.load(loadpath + '%s-Train.npy' % part)
    testData = numpy.load(loadpath + '%s-Test.npy' % part)
    print(numpy.shape(trainData), numpy.shape(testData))
    return trainData, testData


def Loader_SpeechLevel(part):
    pass
