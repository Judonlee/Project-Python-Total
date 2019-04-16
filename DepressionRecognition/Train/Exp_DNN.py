import numpy
from DepressionRecognition.Model.DNN import DNN
from sklearn.preprocessing import scale
import os


def Loader(partname):
    loadpath = 'E:/ProjectData_Depression/SpeechLevel/%s-%s.csv'
    trainData = numpy.genfromtxt(fname=loadpath % (partname, 'Train'), dtype=float, delimiter=',')
    trainLabel = numpy.genfromtxt(fname='E:/ProjectData_Depression/SpeechLevel/TrainLabel.csv', dtype=float,
                                  delimiter=',')
    testData = numpy.genfromtxt(fname=loadpath % (partname, 'Test'), dtype=float, delimiter=',')
    testLabel = numpy.genfromtxt(fname='E:/ProjectData_Depression/SpeechLevel/TestLabel.csv', dtype=float,
                                 delimiter=',')

    totalData = numpy.concatenate([trainData, testData], axis=0)
    totalData = scale(totalData)
    trainData = totalData[0:len(trainData)]
    testData = totalData[len(trainData):]

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
    return trainData, trainLabel, testData, testLabel


if __name__ == '__main__':
    usedpart = 'MA-10-sentence'
    trainData, trainLabel, testData, testLabel = Loader(partname=usedpart)

    savepath = 'E:/ProjectData_Depression/SpeechLevel/'
    os.makedirs(os.path.join(savepath, usedpart))
    classifier = DNN(trainData=trainData, trainLabel=trainLabel)
    for episode in range(1000):
        print('\nEpisode %d Total Loss = %f' % (episode, classifier.Train()))
        classifier.Test(logname=os.path.join(savepath, usedpart, '%04d.csv' % episode), testData=testData,
                        testLabel=testLabel)
