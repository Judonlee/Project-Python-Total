from LIDC_Project.Loader import LoadPCA, LoadDX
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import os
import multiprocessing
import time
import numpy


def Treatment(classifyType):
    loadFeature = 'R=2_P=16'
    for loadType in ['PCA']:
        for componentNumber in range(1, 31):
            for part in range(5):
                savepath = 'E:/ProjectData_LIDC/Features/Step4_Result/%s_%s/%s/' % (loadFeature, loadType, classifyType)
                if os.path.exists(savepath + 'Component%04d-Part%d.csv' % (componentNumber, part)): continue

                if loadType == 'PCA':
                    trainData, trainLabel, testData, testLabel = LoadPCA(name=loadFeature + '_' + loadType, part=part,
                                                                         componentNumber=componentNumber)
                if loadType == 'DX':
                    trainData, trainLabel, testData, testLabel = LoadDX(name=loadFeature + '_' + loadType, part=part,
                                                                        componentNumber=componentNumber)
                print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))

                if not os.path.exists(savepath): os.makedirs(savepath)
                with open(savepath + 'Component%04d-Part%d.csv' % (componentNumber, part), 'w'):
                    pass

                if classifyType == 'SVC': clf = SVC(probability=True)
                if classifyType == 'Gaussian': clf = GaussianNB()
                if classifyType == 'Tree': clf = DecisionTreeClassifier()
                if classifyType == 'AdaBoost': clf = AdaBoostClassifier()

                clf.fit(trainData, trainLabel)

                result = clf.predict_proba(testData)
                with open(savepath + 'Component%04d-Part%d.csv' % (componentNumber, part), 'w') as file:
                    for indexX in range(len(result)):
                        for indexY in range(len(result[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(result[indexX][indexY]))
                        file.write('\n')


if __name__ == '__main__':
    for classifyType in ['SVC', 'Gaussian', 'Tree', 'AdaBoost']:
        processList = []
        for _ in range(1):
            process = multiprocessing.Process(target=Treatment, args=[classifyType])
            processList.append(process)
            process.start()
            time.sleep(5)

        for process in processList:
            process.join()
