import os
import numpy
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint

if __name__ == '__main__':
    loadpath = 'Bands30/'
    savepath = 'Bands30.csv'
    trainData = numpy.load(loadpath + 'TrainData.npy')
    trainLabel = numpy.load(loadpath + 'TrainLabel.npy')
    testData = numpy.load(loadpath + 'TestData.npy')
    testLabel = numpy.load(loadpath + 'TestLabel.npy')
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))

    clf = DecisionTreeClassifier()
    clf.fit(trainData, trainLabel)
    predict = clf.predict(testData)

    matrix = numpy.zeros((5, 5))
    for index in range(len(predict)):
        matrix[int(testLabel[index])][int(predict[index])] += 1
    pprint(matrix)

    with open(savepath, 'w') as file:
        for indexA in range(numpy.shape(matrix)[0]):
            for indexB in range(numpy.shape(matrix)[1]):
                if indexB != 0: file.write(',')
                file.write(str(matrix[indexA][indexB]))
            file.write('\n')
