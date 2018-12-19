from Demo.Loader import Loader
from Demo.Model_LSTM import SingleLSTM
import numpy
import os

APPOINT_SESSION = 1
BANDS = 30
TOTAL_EPISODE = 100

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    loadpath = 'E:/CTC_Target/Features/Bands%d/' % BANDS
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(
        loadpath=loadpath, testSession=APPOINT_SESSION)

    classifier = SingleLSTM(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                            featureShape=numpy.shape(trainData[0])[1])
    for episode in range(TOTAL_EPISODE):
        loss = classifier.Train()

        matrix = classifier.Test(testData=testData, testLabel=testLabel, testSeq=testSeq)
        precision = 0
        for index in range(numpy.shape(matrix)[0]):
            precision += matrix[index][index]
        precision /= numpy.sum(numpy.sum(matrix))

        print('\rEpisode %d Total Loss = %f\tPrecision = %f' % (episode, loss, precision))
