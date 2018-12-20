from sklearn.svm import SVC
import numpy

if __name__ == '__main__':
    loadpath = 'D:/Matlab/VGG19/'
    trainData = numpy.load(loadpath + 'Ohm-Data.npy')
    trainLabel = numpy.load(loadpath + 'Ohm-Label.npy')
    testData = numpy.load(loadpath + 'Mont-Data.npy')
    testLabel = numpy.load(loadpath + 'Mont-Label.npy')

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
    clf = SVC(class_weight={0: 1.1, 1: 0.5, 2: 0.2, 3: 1.5, 4: 1.4})
    clf.fit(trainData, trainLabel)
    print('Train Completed')
    matrix = numpy.zeros((5, 5))
    predict = clf.predict(testData)
    for index in range(len(predict)):
        matrix[testLabel[index]][predict[index]] += 1

    for indexX in range(5):
        for indexY in range(5):
            print(matrix[indexX][indexY], end='\t')
        print()
