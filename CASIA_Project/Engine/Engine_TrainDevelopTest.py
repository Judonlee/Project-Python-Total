import os


def Engine_TrainDevelopTest(dataClass, classifier, savePath, totalEpoch=100):
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    file = open(savePath + 'Information.txt', 'w')
    file.write(classifier.information)
    file.close()

    for episode in range(totalEpoch):
        totalLoss = classifier.Train()
        print('\rEpisode%.4d/%.4d : TotalLoss = %f' % (episode, totalEpoch, totalLoss))

        file = open(savePath + '%.4d.csv' % episode, 'w')
        MAE, RMSE = classifier.Test(testData=dataClass.trainData, testLabel=dataClass.trainLabel)
        file.write(str(MAE) + ',' + str(RMSE) + ',')
        print('Train MAE :', MAE, 'RMSE :', RMSE)

        MAE, RMSE = classifier.Test(testData=dataClass.developData, testLabel=dataClass.developLabel)
        file.write(str(MAE) + ',' + str(RMSE) + ',')
        print('Develop MAE :', MAE, 'RMSE :', RMSE)

        MAE, RMSE = classifier.Test(testData=dataClass.testData, testLabel=dataClass.testLabel)
        file.write(str(MAE) + ',' + str(RMSE))
        print('Test MAE :', MAE, 'RMSE :', RMSE, '\n\n')
        file.close()
