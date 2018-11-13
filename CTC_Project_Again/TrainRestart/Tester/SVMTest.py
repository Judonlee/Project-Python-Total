import numpy
import os
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint

if __name__ == '__main__':
    loadpath = 'D:/GitHub/CTC_Project_Again/TrainRestart/Tester/Bands-30-Session-1/'
    savepath = 'Result/'
    for index in range(100):
        print('Treating :', index)
        if os.path.exists(savepath + '%04d.csv' % index): continue
        if not os.path.exists(loadpath + 'TestData-%d.npy' % index): continue
        file = open(savepath + '%04d.csv' % index, 'w')

        trainData = numpy.load(loadpath + 'TrainData-%d.npy' % index)
        trainLabel = numpy.load(loadpath + 'TrainLabel-%d.npy' % index)
        trainGroundLabel = numpy.load(loadpath + 'TrainGroundLabel.npy')
        trainSeq = numpy.load(loadpath + 'TrainSeq.npy')

        testData = numpy.load(loadpath + 'TestData-%d.npy' % index)
        testLabel = numpy.load(loadpath + 'TestLabel-%d.npy' % index)
        testGroundLabel = numpy.load(loadpath + 'TestGroundLabel.npy')
        testSeq = numpy.load(loadpath + 'TestSeq.npy')

        trainLabel = numpy.argmax(trainLabel, axis=1)
        testLabel = numpy.argmax(testLabel, axis=1)
        print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
        clf = DecisionTreeClassifier()
        clf.fit(trainData, trainLabel)

        matrix = numpy.zeros((4, 4))
        predict = clf.predict(testData)

        startPosition = 0
        for index in range(len(testGroundLabel)):
            labelList = predict[startPosition:startPosition + testSeq[index]]
            saver = numpy.zeros(5)
            for counter in labelList:
                saver[counter] += 1
            startPosition += testSeq[index]

            matrix[numpy.argmax(numpy.array(testGroundLabel[index]))][numpy.argmax(numpy.array(saver[1:]))] += 1
        print(matrix)

        for indexX in range(len(matrix)):
            for indexY in range(len(matrix[indexX])):
                if indexY != 0: file.write(',')
                file.write(str(matrix[indexX][indexY]))
            file.write('\n')
        file.close()
