from CTC_Project.Module.LSTM_R_FinalPooling import LSTM_FinalPooling
import numpy
import tensorflow

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\Emergency\\40\\'
    trainData = numpy.load(loadpath + 'TrainData.npy')
    trainLabel = numpy.load(loadpath + 'TrainLabel.npy')
    trainnSeq = numpy.load(loadpath + 'TrainSeq.npy')
    developData = numpy.load(loadpath + 'DevelopData.npy')
    developLabel = numpy.load(loadpath + 'DevelopLabel.npy')
    developSeq = numpy.load(loadpath + 'DevelopSeq.npy')
    testData = numpy.load(loadpath + 'TestData.npy')
    testLabel = numpy.load(loadpath + 'TestLabel.npy')
    testSeq = numpy.load(loadpath + 'TestSeq.npy')

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainnSeq))
    print(numpy.shape(developData), numpy.shape(developLabel), numpy.shape(developSeq))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testLabel))

    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = LSTM_FinalPooling(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainnSeq,
                                       featureShape=40, batchSize=8)
        for episode in range(100):
            loss = classifier.Train()
            print('\r', end='')
            print('Episode', episode, '\tLoss :', loss)
            trainMAE, trainRMSE = classifier.Test(testData=trainData, testLabel=trainLabel, testSequence=trainnSeq)
            developMAE, developRMSE = classifier.Test(testData=developData, testLabel=developLabel,
                                                      testSequence=developSeq)
            testMAE, testRMSE = classifier.Test(testData=testData, testLabel=testLabel, testSequence=testSeq)
            print(trainMAE, '\t', trainRMSE, '\t', developMAE, '\t', developRMSE, '\t', testMAE, '\t', testRMSE)
