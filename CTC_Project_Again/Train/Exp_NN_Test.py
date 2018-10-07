import tensorflow
from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_LLD_Loader
from CTC_Project_Again.Model.DNN_Layer4 import DNN
import numpy

if __name__ == '__main__':
    appoint = 9
    loadpath = 'D:\\ProjectData\\IEMOCAP\\IEMOCAP-Features\\eGeMAPS-Singe-Normalized\\'
    trainData, trainLabel, testData, testLabel = IEMOCAP_LLD_Loader(loadpath=loadpath, appoint=appoint)
    classifier = DNN(trainData=trainData, trainLabel=trainLabel, featureShape=numpy.shape(trainData)[1],
                     numClass=numpy.shape(trainLabel)[1], learningRate=1e-4)
    print(classifier.information)

    WATrace, UATrace = [], []
    for episode in range(100):
        print('Episode', episode)
        classifier.Train()
        matrix = classifier.Test(testData=testData, testLabel=testLabel)

        print()
        for sample in matrix:
            for subsample in sample:
                print(subsample, end=',')
            print()
        print()

        WA = 0
        for index in range(len(matrix)):
            WA += matrix[index][index]
        WA /= len(testData)

        UA = 0
        for index in range(len(matrix)):
            UA += matrix[index][index] / sum(matrix[index])
        UA /= len(matrix)
        print('WA =', WA, 'UA =', UA)
        WATrace.append(WA)
        UATrace.append(UA)

    for index in range(len(WATrace)):
        print(WATrace[index], ',', UATrace[index])
    print(max(WATrace), max(UATrace))
