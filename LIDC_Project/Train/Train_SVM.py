from LIDC_Project.Loader.LIDC_Loader import LIDC_Loader_Npy
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import numpy
import os
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.preprocessing import scale

if __name__ == '__main__':
    for appoint in range(10):
        # savepath = 'E:/LIDC/Result-SVM/OriginCsv/Appoint-%d/' % appoint
        # if not os.path.exists(savepath):
        #     os.makedirs(savepath)

        trainData, trainLabel, testData, testLabel = LIDC_Loader_Npy(
            loadpath='E:/LIDC/Npy/LBP_P=24_R=3/Appoint-%d/' % appoint)

        totalNumber = numpy.sum(testLabel, axis=0)[0]

        trainData = numpy.reshape(trainData, newshape=[-1, 4096])
        testData = numpy.reshape(testData, newshape=[-1, 4096])
        trainLabel = numpy.argmax(trainLabel, axis=1)
        testLabel = numpy.argmax(testLabel, axis=1)
        # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
        # print(trainData[0:5])
        totalData = numpy.concatenate((trainData, testData), axis=0)
        pca = PCA(n_components=10)
        pca.fit(totalData)

        trainData = pca.transform(trainData)
        testData = pca.transform(testData)

        totalData = numpy.concatenate((trainData, testData), axis=0)
        totalData = scale(totalData)
        trainData = totalData[0:len(trainData)]
        testData = totalData[len(trainData):]
        # print(trainData[0:5])

        clf = GaussianNB()
        clf.fit(trainData, trainLabel)
        # joblib.dump(clf, savepath + 'Parameter.m')
        result = clf.predict(testData)
        # probability = clf.predict_proba(testData)
        # print(probability[0:10])
        # exit()
        matrix = numpy.zeros((2, 2))
        for index in range(len(result)):
            matrix[testLabel[index]][result[index]] += 1
        print(appoint)
        for indexX in range(2):
            for indexY in range(2):
                if indexY != 0: print(',', end='')
                print(matrix[indexX][indexY], end='')
            print()

        # for threshold in range(101):
        #     counter = 0
        #     for index in range(len(probability)):
        #         if testLabel[index] == 0 and probability[index][0] <= threshold / 100:
        #             counter += 1
        #     print(counter / totalNumber)

# 1020.0,127.0,
# 129.0,1664.0,
