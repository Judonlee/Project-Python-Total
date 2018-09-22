import os
from time import strftime


def TrainTestEngine(dataClass, classifier, totalEpoch, savepath):
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    print(classifier.information)

    file = open(savepath + 'Information.txt', 'w')
    file.write(classifier.information)
    file.close()

    for episode in range(totalEpoch):
        file = open(savepath + '%04d.txt' % episode, 'w')
        loss = classifier.Train()
        print('\rEpisode %d/%d Total Loss : %f' % (episode, 100, loss) + strftime("%Y/%m/%d %H:%M:%S"))
        print('Train Part :')
        matrix = classifier.Test(testData=dataClass.trainData, testLabel=dataClass.trainLabel,
                                 testSeq=dataClass.trainSeq)
        print(matrix)

        file.write('Train Part :\n')
        for indexX in range(len(matrix)):
            for indexY in range(len(matrix[indexX])):
                if indexY != 0: file.write(',')
                file.write(str(matrix[indexX][indexY]))
            file.write('\n')

        print('Test Part :')
        matrix = classifier.Test(testData=dataClass.testData, testLabel=dataClass.testLabel, testSeq=dataClass.testSeq)
        print(matrix)
        print('\n')

        file.write('\n\nTest Part :\n')
        for indexX in range(len(matrix)):
            for indexY in range(len(matrix[indexX])):
                if indexY != 0: file.write(',')
                file.write(str(matrix[indexX][indexY]))
            file.write('\n')

        file.close()

    classifier.Save(savepath=savepath + 'NeuralParameter')
