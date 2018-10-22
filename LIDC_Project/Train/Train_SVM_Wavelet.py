from LIDC_Project.Loader.LIDC_Loader import LIDC_Loader_Npy
from sklearn.svm import SVC
import numpy
import os
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import scale

if __name__ == '__main__':
    for appoint in range(10):
        part = ['cA', 'cD', 'cH', 'cV']
        # part = ['cV']
        conf = 'db4'
        trainData, testData = [], []
        trainLabel = numpy.load('E:/LIDC/Npy/Wavelet-' + conf + '/Appoint-%d/TrainLabel.npy' % appoint)
        testLabel = numpy.load('E:/LIDC/Npy/Wavelet-' + conf + '/Appoint-%d/TestLabel.npy' % appoint)
        trainLabel = numpy.argmax(trainLabel, axis=1)
        testLabel = numpy.argmax(testLabel, axis=1)

        for sample in part:
            currentData = numpy.load('E:/LIDC/Npy/Wavelet-' + conf + '/Appoint-%d/' % appoint + sample + '-Train.npy')
            currentData = numpy.reshape(currentData,
                                        newshape=[-1, numpy.shape(currentData)[1] * numpy.shape(currentData)[2]])
            if len(trainData) == 0:
                trainData = currentData.copy()
            else:
                trainData = numpy.concatenate((trainData, currentData), axis=1)

            currentData = numpy.load('E:/LIDC/Npy/Wavelet-' + conf + '/Appoint-%d/' % appoint + sample + '-Test.npy')
            currentData = numpy.reshape(currentData,
                                        newshape=[-1, numpy.shape(currentData)[1] * numpy.shape(currentData)[2]])
            if len(testData) == 0:
                testData = currentData.copy()
            else:
                testData = numpy.concatenate((testData, currentData), axis=1)

        print(numpy.shape(trainData), numpy.shape(testData), numpy.shape(trainLabel), numpy.shape(testLabel))

        totalData = numpy.concatenate((trainData, testData), axis=0)
        pca = PCA(n_components=10)
        pca.fit(totalData)
        print(sum(pca.explained_variance_ratio_))

        trainData = pca.transform(trainData)
        testData = pca.transform(testData)

        totalData = numpy.concatenate((trainData, testData), axis=0)
        totalData = scale(totalData)
        trainData = totalData[0:len(trainData)]
        testData = totalData[len(trainData):]
        clf = SVC()
        clf.fit(trainData, trainLabel)
        result = clf.predict(testData)

        matrix = numpy.zeros((2, 2))
        for index in range(len(result)):
            matrix[testLabel[index]][result[index]] += 1
        print(conf, part, appoint)
        for indexX in range(2):
            for indexY in range(2):
                if indexY != 0: print(',', end='')
                print(matrix[indexX][indexY], end='')
            print()
