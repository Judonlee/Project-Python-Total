import numpy
import os


def Load_EncoderDecoder():
    loadpath = 'E:/ProjectData_Depression/Step5_Assembly/'
    trainData, trainLabel, trainDataSeq, trainLabelSeq, testData, testLabel, testDataSeq, testLabelSeq = [], [], [], [], [], [], [], []

    for indexA in ['Train', 'Develop', 'Test']:
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
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
    loadpath = 'D:/ProjectData/AVEC2017-Bands40/Step5_Assembly/'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []


if __name__ == '__main__':
    trainData, trainLabel, trainDataSeq, trainLabelSeq, testData, testLabel, testDataSeq, testLabelSeq = Load_EncoderDecoder()
    print(trainData[0:5])
    print(trainLabel[0:5])
    print(trainDataSeq[0:5])
    print(trainLabelSeq[0:5])
