from LIDC_Project.Loader.LIDC_Loader import LIDC_Loader_Npy
from sklearn.svm import SVC
import numpy
import os
from sklearn.externals import joblib

if __name__ == '__main__':
    appoint = 0
    savepath = 'E:/LIDC/Result-SVM/OriginCsv/Appoint-%d/' % appoint
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    trainData, trainLabel, testData, testLabel = LIDC_Loader_Npy(loadpath='E:/LIDC/Npy/OriginCsv/Appoint-%d/' % appoint)
    trainData = numpy.reshape(trainData, newshape=[-1, 4096])
    testData = numpy.reshape(testData, newshape=[-1, 4096])
    trainLabel = numpy.argmax(trainLabel, axis=1)
    testLabel = numpy.argmax(testLabel, axis=1)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))

    clf = SVC()
    clf.fit(trainData, trainLabel)
    joblib.dump(clf, savepath + 'Parameter.m')
    result = clf.predict(testData)
    matrix = numpy.zeros((2, 2))
    for index in range(len(result)):
        matrix[testLabel[index]][result[index]] += 1
    for sample in matrix:
        for subsample in matrix:
            print(subsample, end=',')
        print()
