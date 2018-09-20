import numpy
import os

if __name__ == '__main__':
    loadpath = 'F:\\AVEC-Final\\TimeThreshold12S\\Features-Normalized\\'
    labelpath = 'F:\\AVEC-Final\\Labels\\'
    savepath = 'F:\\AVEC-Final\\TimeThreshold12S\\Features-Npy\\'

    for indexA in os.listdir(loadpath):
        os.makedirs(savepath + indexA)
        trainData, trainLabel, developData, developLabel, testData, testLabel = [], [], [], [], [], []
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    print(indexA, indexB, indexC, indexD)

                    currentData = numpy.genfromtxt(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD,
                                                   dtype=float, delimiter=',')

                    if len(numpy.shape(currentData)) < 2:
                        currentData = currentData[numpy.newaxis, :]

                    currentLabel = numpy.genfromtxt(labelpath + indexC + '\\' + indexD[0:6] + 'Depression.csv',
                                                    dtype=int, delimiter=',')
                    print(numpy.shape(currentData), numpy.shape(currentLabel))

                    if indexC == 'Train':
                        trainData.append(currentData)
                        trainLabel.append(currentLabel)
                    if indexC == 'Develop':
                        developData.append(currentData)
                        developLabel.append(currentLabel)
                    if indexC == 'Test':
                        testData.append(currentData)
                        testLabel.append(currentLabel)
        print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(developData), numpy.shape(developLabel),
              numpy.shape(testData), numpy.shape(testLabel))
        numpy.save(savepath + indexA + '\\TrainData.npy', trainData)
        numpy.save(savepath + indexA + '\\TrainLabel.npy', trainLabel)
        numpy.save(savepath + indexA + '\\DevelopData.npy', developData)
        numpy.save(savepath + indexA + '\\DevelopLabel.npy', developLabel)
        numpy.save(savepath + indexA + '\\TestData.npy', testData)
        numpy.save(savepath + indexA + '\\TestLabel.npy', testLabel)
        # break
