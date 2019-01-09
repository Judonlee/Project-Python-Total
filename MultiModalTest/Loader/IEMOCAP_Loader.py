import os
import numpy


def LoaderLeaveOneSession(loadpath, appointSession):
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            data = numpy.load(os.path.join(loadpath, '%s-Session%d-Data.npy' % (gender, session)))
            label = numpy.load(os.path.join(loadpath, '%s-Session%d-Label.npy' % (gender, session)))

            seq = []
            for sample in data:
                seq.append(len(sample))

            if session == appointSession:
                testData.extend(data)
                testLabel.extend(label)
                testSeq.extend(seq)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(testData),
          numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


def LoaderLeaveOneSpeaker(loadpath, appointSession, appointGender):
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            data = numpy.load(os.path.join(loadpath, '%s-Session%d-Data.npy' % (gender, session)))
            label = numpy.load(os.path.join(loadpath, '%s-Session%d-Label.npy' % (gender, session)))

            seq = []
            for sample in data:
                seq.append(len(sample))

            if session == appointSession and gender == appointGender:
                testData.extend(data)
                testLabel.extend(label)
                testSeq.extend(seq)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
                trainSeq.extend(seq)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(testData),
          numpy.shape(testLabel), numpy.shape(testSeq))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


def CNNLoaderLeaveOneSpeaker(loadpath, appointSession, appointGender, commonSize=500):
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            data = numpy.load(os.path.join(loadpath, '%s-Session%d-Data.npy' % (gender, session)))
            label = numpy.load(os.path.join(loadpath, '%s-Session%d-Label.npy' % (gender, session)))

            for index in range(numpy.shape(data)[0]):
                # print(numpy.shape(data[index]))
                if numpy.shape(data[index])[0] < 500:
                    data[index] = numpy.concatenate(
                        (data[index],
                         numpy.zeros([commonSize - numpy.shape(data[index])[0], numpy.shape(data[index])[1]])),
                        axis=0)
                else:
                    data[index] = data[index][0:500]
                # print(numpy.shape(data[index]))

            if session == appointSession and gender == appointGender:
                testData.extend(data)
                testLabel.extend(label)
            else:
                trainData.extend(data)
                trainLabel.extend(label)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0), numpy.shape(testData),
          numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


if __name__ == '__main__':
    # trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSpeaker(
    #     loadpath='D:/ProjectData/Features/IEMOCAP-Npy/Bands30/', appointSession=1, appointGender='Female')
    # for sample in trainLabel:
    #     if len(sample) == 0: print(sample)
    trainData, trainLabel, testData, testLabel = CNNLoaderLeaveOneSpeaker(
        loadpath='E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands30-CNN/', appointSession=1,
        appointGender='Female')
