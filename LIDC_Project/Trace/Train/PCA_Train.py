from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from LIDC_Project.Loader.LIDC_Loader import LIDC_NewLoader
from LIDC_Project.Trace.Train.Tools import LoaderPCA
import numpy
import os

USED_PART_A = 'Origin-Npy'
USED_PART_B = 'LBP-Npy\\R=1_P=4'
CHOOSED_CLASSIFIER = 'Tree'
# SVM    Tree    Gaussian    AdaBoost

if __name__ == '__main__':
    loadpathLeft = 'D:/LIDC/%s/' % USED_PART_A
    loadpathRight = 'D:/LIDC/%s/' % USED_PART_B
    savepath = 'D:/LIDC/Result/PCA-%s/%s/' % (CHOOSED_CLASSIFIER, USED_PART_B)
    if not os.path.exists(savepath): os.makedirs(savepath)

    for pcaEpisode in range(1, 200):
        for part in range(5):
            if os.path.exists(savepath + 'Episode%04d-%d.csv' % (pcaEpisode, part)): continue

            with open(savepath + 'Episode%04d-%d.csv' % (pcaEpisode, part), 'w'):
                pass

            trainData, trainLabel, testData, testLabel = LoaderPCA(leftPath=loadpathLeft, rightPath=loadpathRight,
                                                                   part=part, pcaEpisode=pcaEpisode)

            if CHOOSED_CLASSIFIER == 'SVM': clf = SVC(probability=True)
            if CHOOSED_CLASSIFIER == 'Tree': clf = DecisionTreeClassifier()
            if CHOOSED_CLASSIFIER == 'Gaussian': clf = GaussianNB()
            if CHOOSED_CLASSIFIER == 'AdaBoost': clf = AdaBoostClassifier()
            clf.fit(trainData, trainLabel)
            predict = clf.predict_proba(testData)

            with open(savepath + 'Episode%04d-%d.csv' % (pcaEpisode, part), 'w') as file:
                for indexX in range(len(predict)):
                    for indexY in range(len(predict[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(predict[indexX][indexY]))
                    file.write('\n')
