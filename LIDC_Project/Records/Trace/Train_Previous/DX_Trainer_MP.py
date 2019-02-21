from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from LIDC_Project.Trace.Train_Previous.Tools import LoadPart, DX_Appoint, MP_Treatment
import numpy
import os


def Treatment():
    used = 'OriginCsv'
    classifier = 'SVM'

    weights = numpy.genfromtxt('WeightResult.csv', dtype=float, delimiter=',')

    # SVM    Tree    Gaussian    AdaBoost
    for pcaPart in range(1, 200):
        savepath = 'E:/LIDC/TreatmentTrace/Step8-Result/DX_Score/%s-%s-%04d/' % (used, classifier, pcaPart)

        if os.path.exists(savepath): continue
        os.makedirs(savepath)
        for appoint in range(10):
            trainData, trainLabel, testData, testLabel = LoadPart(
                loadpath='E:/LIDC/TreatmentTrace/Step7-TotalNpy/%s/' % used,
                appoint=appoint)
            trainData = numpy.reshape(trainData, [-1, numpy.shape(trainData)[1] * numpy.shape(trainData)[2]])
            trainLabel = numpy.argmax(trainLabel, axis=1)
            testData = numpy.reshape(testData, [-1, numpy.shape(testData)[1] * numpy.shape(testData)[2]])
            testLabel = numpy.argmax(testLabel, axis=1)

            trainData, testData = DX_Appoint(trainData=trainData, testData=testData, featureShape=pcaPart,
                                             weights=weights)

            print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
            # exit()
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
    MP_Treatment(function=Treatment, times=5)
