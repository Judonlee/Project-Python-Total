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


if __name__ == '__main__':
    loadpath = 'D:/ProjectData/CTC_Target/Features/Bands30/'
    Load_Part(loadpath=loadpath, appointGender='Female', appointSession=1)
