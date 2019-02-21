from LIDC_Project.Loader import LoadPCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import os
import multiprocessing
import time


def Treatment(classifyType):
    for componentNumber in range(1, 201):
        for part in range(5):
            savepath = 'E:/ProjectData_LIDC/Features/Step4_Result/%s/' % classifyType
            if os.path.exists(savepath + 'Component%04d-Part%d.csv' % (componentNumber, part)): continue
            trainData, trainLabel, testData, testLabel = LoadPCA(part=part, componentNumber=componentNumber)

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
    classifyType = 'SVC'
    # SVC,Gaussian,Tree,AdaBoost

    processList = []
    for _ in range(2):
        process = multiprocessing.Process(target=Treatment, args=[classifyType])
        processList.append(process)
        process.start()
        time.sleep(5)

    for process in processList:
        process.join()
