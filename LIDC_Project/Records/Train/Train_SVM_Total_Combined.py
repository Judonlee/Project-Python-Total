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

    loadpath = 'E:/LIDC/Npy/'
    for appoint in range(10):
        totalTrainData, totalTrainLabel, totalTestData, totalTestLabel = [], [], [], []
        for part in ['Wavelet-db1', 'Wavelet-db2', 'Wavelet-db4']:
            for name in ['cA', 'cD', 'cH', 'cV']:
                partTrainData = numpy.load(loadpath + part + '/Appoint-%d/%s-Train.npy' % (appoint, name))
                partTestData = numpy.load(loadpath + part + '/Appoint-%d/%s-Test.npy' % (appoint, name))
                partTrainData = numpy.reshape(partTrainData,
                                              (-1, numpy.shape(partTrainData)[1] * numpy.shape(partTrainData)[2]))
                partTestData = numpy.reshape(partTestData,
                                             (-1, numpy.shape(partTestData)[1] * numpy.shape(partTestData)[2]))
                # print(numpy.shape(partTrainData), numpy.shape(partTestData))
                if len(totalTrainData) == 0:
                    totalTrainData = partTrainData.copy()
                    totalTestData = partTestData.copy()
                    totalTrainLabel = numpy.load(loadpath + part + '/Appoint-%d/TrainLabel.npy' % appoint)
                    totalTrainLabel = numpy.argmax(totalTrainLabel, axis=1)
                    totalTestLabel = numpy.load(loadpath + part + '/Appoint-%d/TestLabel.npy' % appoint)
                    totalTestLabel = numpy.argmax(totalTestLabel, axis=1)
                else:
                    totalTrainData = numpy.concatenate((totalTrainData, partTrainData), axis=1)
                    totalTestData = numpy.concatenate((totalTestData, partTestData), axis=1)
        for name in ['LBP_P=4_R=1', 'LBP_P=8_R=1', 'LBP_P=16_R=2', 'LBP_P=24_R=3']:
            partTrainData, partTrainLabel, partTestData, partTestLabel = LIDC_Loader_Npy(
                loadpath='E:/LIDC/Npy/%s/Appoint-%d/' % (name, appoint))
            partTrainData = numpy.reshape(partTrainData,
                                          (-1, numpy.shape(partTrainData)[1] * numpy.shape(partTrainData)[2]))
            partTestData = numpy.reshape(partTestData,
                                         (-1, numpy.shape(partTestData)[1] * numpy.shape(partTestData)[2]))
            totalTrainData = numpy.concatenate((totalTrainData, partTrainData), axis=1)
            totalTestData = numpy.concatenate((totalTestData, partTestData), axis=1)
        partTrainData, partTrainLabel, partTestData, partTestLabel = LIDC_Loader_Npy(
            loadpath='E:/LIDC/Npy/OriginCsv/Appoint-%d/' % appoint)
        partTrainData = numpy.reshape(partTrainData,
                                      (-1, numpy.shape(partTrainData)[1] * numpy.shape(partTrainData)[2]))
        partTestData = numpy.reshape(partTestData,
                                     (-1, numpy.shape(partTestData)[1] * numpy.shape(partTestData)[2]))
        totalTrainData = numpy.concatenate((totalTrainData, partTrainData), axis=1)
        totalTestData = numpy.concatenate((totalTestData, partTestData), axis=1)

        print('\n\n\nPart %d Load Completed' % appoint, numpy.shape(totalTrainData), numpy.shape(totalTrainLabel),
              numpy.shape(totalTestData), numpy.shape(totalTestLabel))
        # exit()
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
    # plt.title('Total-Assembly')
    # plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    # plt.legend()
    # plt.show()
    #
    # for sample in aucList:
    #     print(sample)
