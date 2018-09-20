from sklearn import svm
import numpy
import os

if __name__ == '__main__':
    conf = 'IS12'
    loadpath = 'F:\\DataAugment20dB-Features-Npy\\' + conf + '\\'
    savepath = 'F:\\AVEC-SVR-Changed\\' + conf + '\\'
    os.makedirs(savepath)
    trainData = numpy.load(loadpath + 'TrainData.npy')
    trainLabel = numpy.load(loadpath + 'TrainLabel.npy')
    developData = numpy.load(loadpath + 'DevelopData.npy')
    developLabel = numpy.load(loadpath + 'DevelopLabel.npy')
    testData = numpy.load(loadpath + 'TestData.npy')
    testLabel = numpy.load(loadpath + 'TestLabel.npy')

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(developData), numpy.shape(developLabel),
          numpy.shape(testData), numpy.shape(testLabel))

    for C in [1, 0.9, 0.8, 0.7, 0.6, 0.5]:
        for epsilon in [0.5, 0.2, 0.1, 0.05, 0.01]:
            for gamma in [0.01, 0.005, 0.001, 0.0005, 0.0001]:
                for tol in [1e-2, 1e-3, 1e-4]:
                    file = open(
                        savepath + 'C=' + str(C) + '-epsilon=' + str(epsilon) + '-gamma=' + str(gamma) + '-tol=' + str(
                            tol) + '.csv', 'w')

                    print('C=' + str(C) + '-epsilon=' + str(epsilon) + '-gamma=' + str(gamma) + '-tol=' + str(tol))
                    clf = svm.SVR(C=C, epsilon=epsilon, gamma=gamma, tol=tol)
                    clf.fit(trainData, trainLabel)

                    predict = clf.predict(developData)
                    MAE, RMSE = 0, 0
                    for index in range(len(predict)):
                        MAE += numpy.abs(predict[index] - developLabel[index])
                        RMSE += (predict[index] - developLabel[index]) * (predict[index] - developLabel[index])
                    MAE = MAE / len(predict)
                    RMSE = numpy.sqrt(RMSE / len(predict))
                    print('Develop Part\t', MAE, '\t', RMSE)
                    file.write(str(MAE) + ',' + str(RMSE) + ',')

                    predict = clf.predict(testData)
                    MAE, RMSE = 0, 0
                    for index in range(len(predict)):
                        MAE += numpy.abs(predict[index] - testLabel[index])
                        RMSE += (predict[index] - testLabel[index]) * (predict[index] - testLabel[index])
                    MAE = MAE / len(predict)
                    RMSE = numpy.sqrt(RMSE / len(predict))
                    print('Test Part\t', MAE, '\t', RMSE)
                    file.write(str(MAE) + ',' + str(RMSE))

                    file.close()
