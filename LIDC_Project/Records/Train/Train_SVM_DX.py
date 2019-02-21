from LIDC_Project.Records.Loader.LIDC_Loader import LIDC_Loader_Npy
from sklearn.ensemble import AdaBoostClassifier
import numpy

if __name__ == '__main__':
    # part = ['OriginCsv', 'LBP_P=4_R=1', 'LBP_P=8_R=1', 'LBP_P=16_R=2', 'LBP_P=24_R=3']
    part = 'LBP_P=24_R=3'
    for appoint in range(10):
        trainData, trainLabel, testData, testLabel = LIDC_Loader_Npy(
            loadpath='E:/LIDC/Npy/DXSelected/' + part + '/Appoint-%d/' % appoint)

        clf = AdaBoostClassifier()

        clf.fit(trainData, trainLabel)
        result = clf.predict(testData)

        matrix = numpy.zeros((2, 2))
        for index in range(len(result)):
            matrix[testLabel[index]][result[index]] += 1
        print(appoint)
        for indexX in range(2):
            for indexY in range(2):
                if indexY != 0: print(',', end='')
                print(matrix[indexX][indexY], end='')
            print()
