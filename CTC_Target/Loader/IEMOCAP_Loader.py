import numpy


def Load(loadpath, appoint):
    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = [], [], [], [], [], [], [], []
    for session in range(1, 6):
        for gender in ['Female', 'Male']:
            data = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
            label = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
            seq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))
            scription = numpy.load(loadpath + '%s-Session%d-Transcription.npy' % (gender, session))

            if session == appoint:
                testData.extend(data)
                testlabel.extend(label)
                testSeq.extend(seq)
                testScription.extend(scription)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
                trainScription.extend(scription)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainScription),
          numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testlabel), numpy.shape(testSeq), numpy.shape(testScription),
          numpy.sum(testlabel, axis=0))
    return trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription


def LoadSpecialLabel(loadpath, appoint, transcriptionpath):
    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = [], [], [], [], [], [], [], []
    for session in range(1, 6):
        for gender in ['Female', 'Male']:
            data = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
            label = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
            seq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))
            scription = numpy.load(transcriptionpath + '%s-Session%d-Transcription.npy' % (gender, session))

            if session == appoint:
                testData.extend(data)
                testlabel.extend(label)
                testSeq.extend(seq)
                testScription.extend(scription)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
                trainScription.extend(scription)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainScription),
          numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testlabel), numpy.shape(testSeq), numpy.shape(testScription),
          numpy.sum(testlabel, axis=0))
    return trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription


def Load_Part(loadpath, appointGender, appointSession):
    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = [], [], [], [], [], [], [], []
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            data = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
            label = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
            seq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))
            scription = numpy.load(loadpath + '%s-Session%d-Transcription.npy' % (gender, session))

            if gender == appointGender and session == appointSession:
                testData.extend(data)
                testlabel.extend(label)
                testSeq.extend(seq)
                testScription.extend(scription)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
                trainScription.extend(scription)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainScription),
          numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testlabel), numpy.shape(testSeq), numpy.shape(testScription),
          numpy.sum(testlabel, axis=0))
    return trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription


def Load_FAU(loadpath):
    trainData = numpy.load(loadpath + 'Ohm-Data.npy')
    trainLabel = numpy.load(loadpath + 'Ohm-Label.npy')
    trainSeq = numpy.load(loadpath + 'Ohm-Seq.npy')
    trainTranscription = numpy.load(loadpath + 'Ohm-Transcription.npy')
    testData = numpy.load(loadpath + 'Mont-Data.npy')
    testLabel = numpy.load(loadpath + 'Mont-Label.npy')
    testSeq = numpy.load(loadpath + 'Mont-Seq.npy')
    testTranscription = numpy.load(loadpath + 'Mont-Transcription.npy')
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainTranscription),
          numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.shape(testTranscription),
          numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, trainSeq, trainTranscription, testData, testLabel, testSeq, testTranscription


def Load_FAU_Transcription(loadpath, transcriptionpath):
    trainData = numpy.load(loadpath + 'Ohm-Data.npy')
    trainLabel = numpy.load(loadpath + 'Ohm-Label.npy')
    trainSeq = numpy.load(loadpath + 'Ohm-Seq.npy')
    trainTranscription = numpy.load(transcriptionpath + 'Ohm-Transcription.npy')
    testData = numpy.load(loadpath + 'Mont-Data.npy')
    testLabel = numpy.load(loadpath + 'Mont-Label.npy')
    testSeq = numpy.load(loadpath + 'Mont-Seq.npy')
    testTranscription = numpy.load(transcriptionpath + 'Mont-Transcription.npy')
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainTranscription),
          numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.shape(testTranscription),
          numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, trainSeq, trainTranscription, testData, testLabel, testSeq, testTranscription


def Load_MSP(loadpath, appointSession):
    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = [], [], [], [], [], [], [], []
    for session in range(1, 7):
        for gender in ['F', 'M']:
            data = numpy.load(loadpath + 'Session%d-%s-Data.npy' % (session, gender))
            label = numpy.load(loadpath + 'Session%d-%s-Label.npy' % (session, gender))
            seq = numpy.load(loadpath + 'Session%d-%s-Seq.npy' % (session, gender))
            scription = numpy.load(loadpath + 'Session%d-%s-Transcription.npy' % (session, gender))

            if session == appointSession:
                testData.extend(data)
                testlabel.extend(label)
                testSeq.extend(seq)
                testScription.extend(scription)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
                trainScription.extend(scription)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainScription),
          numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testlabel), numpy.shape(testSeq), numpy.shape(testScription),
          numpy.sum(testlabel, axis=0))
    return trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription


def Load_MSP_Part(loadpath, appointGender, appointSession):
    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = [], [], [], [], [], [], [], []
    for gender in ['F', 'M']:
        for session in range(1, 7):
            data = numpy.load(loadpath + 'Session%d-%s-Data.npy' % (session, gender))
            label = numpy.load(loadpath + 'Session%d-%s-Label.npy' % (session, gender))
            seq = numpy.load(loadpath + 'Session%d-%s-Seq.npy' % (session, gender))
            scription = numpy.load(loadpath + 'Session%d-%s-Transcription.npy' % (session, gender))

            if gender == appointGender and session == appointSession:
                testData.extend(data)
                testlabel.extend(label)
                testSeq.extend(seq)
                testScription.extend(scription)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
                trainScription.extend(scription)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainScription),
          numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testlabel), numpy.shape(testSeq), numpy.shape(testScription),
          numpy.sum(testlabel, axis=0))
    return trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription


if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/Feature/Bands-30/'
    Load_MSP(loadpath=loadpath, appointSession=1)
