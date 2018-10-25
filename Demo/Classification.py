from sklearn.svm import SVC
from sklearn.preprocessing import scale
import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-OpenSmile/IS09/'
    appoint = 0

    trainData, trainLabel, testData, testLabel = [], [], [], []
    for indexA in ['improve']:
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                print('Loading :', indexA, indexB, indexC)
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    for indexE in os.listdir(os.path.join(loadpath, indexA, indexB, indexC, indexD)):
                        data = numpy.genfromtxt(os.path.join(loadpath, indexA, indexB, indexC, indexD, indexE),
                                                dtype=float, delimiter=',')
                        if indexD == 'ang': label = 0
                        if indexD == 'exc' or indexD == 'hap': label = 1
                        if indexD == 'neu': label = 2
                        if indexD == 'sad': label = 3
                        if ['Female', 'Male'].index(indexB) * 5 + ['Session1', 'Session2', 'Session3', 'Session4',
                                                                   'Session5'].index(indexC) == appoint:
                            testData.append(data)
                            testLabel.append(label)
                        else:
                            trainData.append(data)
                            trainLabel.append(label)
    print('Load Completed\n')

    totalData = numpy.concatenate((trainData, testData), axis=0)
    totalData = scale(totalData)

    trainData = totalData[0:len(trainData)]
    testData = totalData[len(trainData):]

    print('Normalization Completed\n')

    print('Train Data Shape =', numpy.shape(trainData))
    print('Train Label Shape =', numpy.shape(trainLabel))
    print('Test Data Shape =', numpy.shape(testData))
    print('Test Label Shape =', numpy.shape(testLabel))

    clf = SVC()
    clf.fit(X=trainData, y=trainLabel)
    result = clf.predict(X=testData)

    matrix = numpy.zeros((4, 4))
    for index in range(len(result)):
        matrix[testLabel[index]][result[index]] += 1
    print(matrix)
