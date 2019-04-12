import numpy
import os
from sklearn.svm import SVR


def Load(name):
    trainData = numpy.genfromtxt(
        fname='E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/Normalization/%s-Train.csv' % name,
        dtype=float, delimiter=',')
    trainLabel = numpy.genfromtxt(
        fname='E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/Normalization/TrainLabel.csv', dtype=float,
        delimiter=',')[:, 2].tolist()
    trainLabel.extend(numpy.genfromtxt(
        fname='E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/Normalization/DevelopLabel.csv', dtype=float,
        delimiter=',')[:, 2])
    testData = numpy.genfromtxt(
        fname='E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/Normalization/%s-Test.csv' % name, dtype=float,
        delimiter=',')
    testLabel = numpy.genfromtxt(
        fname='E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/Normalization/TestLabel.csv', dtype=float,
        delimiter=',')[:, 2]
    # print(trainLabel)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
    return trainData, trainLabel, testData, testLabel


if __name__ == '__main__':
    name = 'SA-0-sentence'
    trainData, trainLabel, testData, testLabel = Load(name=name)
    clf = SVR()
    clf.fit(trainData, trainLabel)
    result = clf.predict(testData)

    with open('E:/ProjectData_Depression/Experiment/HierarchyAutoEncoder/%s.csv' % name, 'w') as file:
        for index in range(len(result)):
            file.write(str(result[index]) + ',' + str(testLabel[index]) + '\n')
