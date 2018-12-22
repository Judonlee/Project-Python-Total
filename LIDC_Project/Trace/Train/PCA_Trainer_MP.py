from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from LIDC_Project.Trace.Train.Tools import LoadPart, PCA_Treatment, MP_Treatment
import numpy
import os


def treatment():
    used = 'OriginCsv'
    classifier = 'Tree'
    # SVM    Tree    Gaussian    AdaBoost
    for pcaPart in range(1, 100):
        savepath = 'E:/LIDC-Result/Step8-Result/%s-%s-%04d/' % (used, classifier, pcaPart)

        if not os.path.exists(savepath): os.makedirs(savepath)
        for appoint in range(10):
            if os.path.exists(savepath + 'Batch%d.csv' % appoint): continue
            if not os.path.exists(savepath): exit()
            with open(savepath + 'Batch%d.csv' % appoint, 'w'):
                pass
            print('PCA Part :', pcaPart, appoint)
            trainData, trainLabel, testData, testLabel = LoadPart(
                loadpath='E:/LIDC/TreatmentTrace/Step7-TotalNpy/%s/' % used,
                appoint=appoint)
            trainData = numpy.reshape(trainData, [-1, numpy.shape(trainData)[1] * numpy.shape(trainData)[2]])
            trainLabel = numpy.argmax(trainLabel, axis=1)
            testData = numpy.reshape(testData, [-1, numpy.shape(testData)[1] * numpy.shape(testData)[2]])
            testLabel = numpy.argmax(testLabel, axis=1)

            trainData, testData = PCA_Treatment(trainData=trainData, testData=testData, componentNumber=pcaPart)

            print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))

            if classifier == 'SVM':
                clf = SVC(probability=True)
            if classifier == 'Tree':
                clf = DecisionTreeClassifier()
            if classifier == 'Gaussian':
                clf = GaussianNB()
            if classifier == 'AdaBoost':
                clf = AdaBoostClassifier()
            clf.fit(trainData, trainLabel)
            predict = clf.predict_proba(testData)

            with open(savepath + 'Batch%d.csv' % appoint, 'w') as file:
                for indexX in range(len(predict)):
                    for indexY in range(len(predict[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(predict[indexX][indexY]))
                    file.write('\n')


if __name__ == '__main__':
    MP_Treatment(function=treatment, times=3)
