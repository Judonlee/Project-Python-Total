import numpy


def Loader(loadpath, session):
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []
    for indexA in ['Female', 'Male']:
        for indexB in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
            data = numpy.load(loadpath + '%s-%s-Data.npy' % (indexA, indexB))
            label = numpy.load(loadpath + '%s-%s-Label.npy' % (indexA, indexB))
            seq = numpy.load(loadpath + '%s-%s-Seq.npy' % (indexA, indexB))
            # print(numpy.shape(data), numpy.shape(label), numpy.shape(seq), numpy.sum(label, axis=0))
            if int(indexB[-1]) == session:
                testData.extend(data)
                testLabel.extend(label)
                testSeq.extend(seq)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


def TranscriptionLoader(loadpath, session):
    trainScription, testScription = [], []
    for indexA in ['Female', 'Male']:
        for indexB in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
            scription = numpy.load(loadpath + '%s-%s-Transcription.npy' % (indexA, indexB))
            if int(indexB[-1]) == session:
                testScription.extend(scription)
            else:
                trainScription.extend(scription)
    print(numpy.shape(trainScription), numpy.shape(testScription))
    return trainScription, testScription


def LoaderTotal(loadpath, session):
    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = [], [], [], [], [], [], [], []
    for indexA in ['Female', 'Male']:
        for indexB in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
            data = numpy.load(loadpath + '%s-%s-Data.npy' % (indexA, indexB))
            label = numpy.load(loadpath + '%s-%s-Label.npy' % (indexA, indexB))
            seq = numpy.load(loadpath + '%s-%s-Seq.npy' % (indexA, indexB))
            scription = numpy.load(loadpath + '%s-%s-Scription.npy' % (indexA, indexB))
            # print(numpy.shape(data), numpy.shape(label), numpy.shape(seq), numpy.sum(label, axis=0))
            if int(indexB[-1]) == session:
                testData.extend(data)
                testLabel.extend(label)
                testSeq.extend(seq)
                testScription.extend(scription)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
                trainScription.extend(scription)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.sum(trainLabel, axis=0),
          numpy.shape(trainScription))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.sum(testLabel, axis=0),
          numpy.shape(testScription))
    return trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription


if __name__ == '__main__':
    # Loader(loadpath='D:/ProjectData/Project-CTC-Data/Csv-Npy/Bands30/', session=1)
    # trainScription, testScription = TranscriptionLoader(loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Tran-CMU-Npy/',
    #                                                     session=1)
    # for sample in testScription:
    #     print(sample)
    LoaderTotal(loadpath='D:/ProjectData/IEMOCAP-New/Bands30/', session=1)
