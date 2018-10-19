import os
import numpy


def IEMOCAP_Loader(loadpath, appoint):
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []
    for indexA in ['improve']:
        for indexB in ['Female', 'Male']:
            print('Loading : ', indexA, indexB)
            for indexC in range(1, 6):
                currentData = numpy.load(loadpath + indexA + '/' + indexB + '/Session' + str(indexC) + '-Data.npy')
                currentLabel = numpy.load(loadpath + indexA + '/' + indexB + '/Session' + str(indexC) + '-Label.npy')
                currentSeq = numpy.load(loadpath + indexA + '/' + indexB + '/Session' + str(indexC) + '-Seq.npy')

                if ['Female', 'Male'].index(indexB) * 5 + indexC - 1 == appoint:
                    testData.extend(currentData)
                    testLabel.extend(currentLabel)
                    testSeq.extend(currentSeq)
                else:
                    trainData.extend(currentData)
                    trainLabel.extend(currentLabel)
                    trainSeq.extend(currentSeq)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


def IEMOCAP_TranscriptionLoader(loadpath, appoint):
    trainTranscription, testTranscription = [], []
    for indexA in ['improve']:
        for indexB in os.listdir(loadpath + indexA):
            for indexC in range(1, 6):
                currentTranscription = numpy.load(
                    loadpath + indexA + '/' + indexB + '/Session' + str(indexC) + '.npy')
                if ['Female', 'Male'].index(indexB) * 5 + indexC - 1 == appoint:
                    testTranscription.extend(currentTranscription)
                else:
                    trainTranscription.extend(currentTranscription)
    return trainTranscription, testTranscription


def IEMOCAP_SeqLabelLoader(loadpath):
    trainLabel = numpy.load(loadpath + 'TrainSeqLabel.npy')
    testLabel = numpy.load(loadpath + 'TestSeqLabel.npy')
    return trainLabel, testLabel


def IEMOCAP_LLD_Loader(loadpath, appoint):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for indexA in ['improve', 'script']:
        for indexB in ['Female', 'Male']:
            for indexC in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
                print(indexA, indexB, indexC)
                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        currentData = numpy.genfromtxt(
                            fname=loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')
                        if indexD == 'ang': currentLabel = [1, 0, 0, 0]
                        if indexD == 'exc' or indexD == 'hap': currentLabel = [0, 1, 0, 0]
                        if indexD == 'neu': currentLabel = [0, 0, 1, 0]
                        if indexD == 'sad': currentLabel = [0, 0, 0, 1]

                        if ['Female', 'Male'].index(indexB) * 5 + ['Session1', 'Session2', 'Session3', 'Session4',
                                                                   'Session5'].index(indexC) == appoint:
                            testData.append(currentData)
                            testLabel.append(currentLabel)
                        else:
                            trainData.append(currentData)
                            trainLabel.append(currentLabel)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


def IEMOCAP_Loader_Npy(loadpath):
    [trainData, trainLabel, trainSeq, trainScription] = numpy.load(file=loadpath + 'TrainData.npy')
    [testData, testLabel, testSeq, testScription] = numpy.load(file=loadpath + 'TestData.npy')

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainScription),
          numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.shape(testScription),
          numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription


def IEMOCAP_Transcription_Loader_Npy_New(loadpath):
    trainScription = numpy.load(file=loadpath + 'TrainTranscription.npy')
    testScription = numpy.load(file=loadpath + 'TestTranscription.npy')
    return trainScription, testScription


if __name__ == '__main__':
    trainTranscription, testTranscription = IEMOCAP_TranscriptionLoader(
        loadpath='F:\\Project-CTC-Data\\Transcription-SingleNumber\\', appoint=0)
    print(testTranscription)
    '''
    loadpath = 'F:\\Project-CTC-Data\\Csv\\Bands120\\'
    savepath = 'F:\\Project-CTC-Data\\Npy\\Bands120\\'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                currentData, currentLabel, currentSeq = [], [], []

                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        print(indexA, indexB, indexC, indexD, indexE)
                        treatData = numpy.genfromtxt(
                            loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')

                        if indexD == 'ang': treatLabel = [1, 0, 0, 0]
                        if indexD == 'exc': treatLabel = [0, 1, 0, 0]
                        if indexD == 'hap': treatLabel = [0, 1, 0, 0]
                        if indexD == 'neu': treatLabel = [0, 0, 1, 0]
                        if indexD == 'sad': treatLabel = [0, 0, 0, 1]
                        # print(numpy.shape(treatData))

                        currentData.append(treatData)
                        currentLabel.append(treatLabel)
                        currentSeq.append(len(treatData))

                if not os.path.exists(savepath + indexA + '\\' + indexB):
                    os.makedirs(savepath + indexA + '\\' + indexB)
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '-Data.npy', currentData)
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '-Label.npy', currentLabel)
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '-Seq.npy', currentSeq)
    '''
