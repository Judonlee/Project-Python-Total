from LIDC_Project.Loader.LIDC_Loader import LIDC_Loader_Npy
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
import numpy
import os
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import scale

from LIDC_Project.External.ROCPrinter import ROCPrinter
import matplotlib.pylab as plt

if __name__ == '__main__':
    aucList = []
    for appoint in range(10):
        totalTrainData, totalTrainLabel, totalTestData, totalTestLabel = [], [], [], []
        for name in ['LBP_P=4_R=1', 'LBP_P=8_R=1', 'LBP_P=16_R=2', 'LBP_P=24_R=3']:
            trainData, trainLabel, testData, testLabel = LIDC_Loader_Npy(
                loadpath='E:/LIDC/Npy/LBP_P=24_R=3/Appoint-%d/' % appoint)
            trainData = numpy.reshape(trainData, (-1, numpy.shape(trainData)[1] * numpy.shape(trainData)[2]))
            trainLabel = numpy.argmax(trainLabel, axis=1)
            testData = numpy.reshape(testData, (-1, numpy.shape(testData)[1] * numpy.shape(testData)[2]))
            testLabel = numpy.argmax(testLabel, axis=1)

            print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
            if len(totalTestData) == 0:
                totalTrainData = trainData.copy()
                totalTrainLabel = trainLabel.copy()
                totalTestData = testData.copy()
                totalTestLabel = testLabel.copy()
            else:
                totalTrainData = numpy.concatenate((totalTrainData, trainData), axis=1)
                totalTestData = numpy.concatenate((totalTestData, testData), axis=1)

        print('\n\n\nLoad Completed Part', appoint, numpy.shape(totalTrainData), numpy.shape(totalTrainLabel),
              numpy.shape(totalTestData), numpy.shape(totalTestLabel))

        totalData = numpy.concatenate((totalTrainData, totalTestData), axis=0)
        pca = PCA(n_components=10)
        pca.fit(totalData)

        totalTrainData = pca.transform(totalTrainData)
        totalTestData = pca.transform(totalTestData)

        totalData = numpy.concatenate((totalTrainData, totalTestData), axis=0)
        totalData = scale(totalData)
        totalTrainData = totalData[0:len(totalTrainData)]
        totalTestData = totalData[len(totalTrainData):]
        # print(trainData[0:5])

        # aucList.append(ROCPrinter(trainData=totalTrainData, trainLabel=totalTrainLabel, testData=totalTestData,
        #                           testLabel=totalTestLabel, appoint=appoint))

        print('Treat Completed Part', appoint, numpy.shape(totalTrainData), numpy.shape(totalTrainLabel),
              numpy.shape(totalTestData), numpy.shape(totalTestLabel))

        clf = AdaBoostClassifier()
        clf.fit(totalTrainData, totalTrainLabel)
        result = clf.predict(totalTestData)

        matrix = numpy.zeros((2, 2))
        for index in range(len(result)):
            matrix[totalTestLabel[index]][result[index]] += 1
        print(appoint)
        for indexX in range(2):
            for indexY in range(2):
                if indexY != 0: print(',', end='')
                print(matrix[indexX][indexY], end='')
            print()

    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('LBP-Assembly')
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    # plt.legend()
    # plt.show()
    #
    # for sample in aucList:
    #     print(sample)
