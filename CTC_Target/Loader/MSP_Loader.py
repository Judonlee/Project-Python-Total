import numpy


def Loader(loadpath, appointSession, appointGender):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for session in range(1, 7):
        for gender in ['F', 'M']:
            data = numpy.load(loadpath + 'Session%d-%s-Data.npy' % (session, gender))
            label = numpy.load(loadpath + 'Session%d-%s-Label.npy' % (session, gender))
            print('Session%d-%s' % (session, gender), numpy.shape(data), numpy.shape(label))

            if appointSession == session and appointGender == gender:
                testData.extend(data)
                testLabel.extend(label)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0), numpy.shape(testData),
          numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel
