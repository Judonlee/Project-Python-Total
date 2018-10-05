from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_LLD_Loader
from sklearn.svm import SVC
import numpy

if __name__ == '__main__':
    appoint = 0
    loadpath = 'D:\\ProjectData\\Project-CTC-Data\\Csv-Single-Normalized\\Bands30\\'
    trainData, trainLabel, testData, testLabel = IEMOCAP_LLD_Loader(loadpath=loadpath, appoint=appoint)

    trainLabelPretreatment = []
    for sample in trainLabel:
        trainLabelPretreatment.append(numpy.argmax(numpy.array(sample)))

    clf = SVC()
    clf.fit(trainData, trainLabelPretreatment)
    predict = clf.predict(testData)
    matrix = numpy.zeros((4, 4))
    for index in range(len(predict)):
        matrix[numpy.argmax(numpy.array(testLabel[index]))][predict[index]] += 1
    print(matrix)
