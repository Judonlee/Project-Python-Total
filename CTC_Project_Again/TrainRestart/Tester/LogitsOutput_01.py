from CTC_Project_Again.Loader.IEMOCAP_Loader_New import LoaderTotal
import tensorflow
from CTC_Project_Again.ModelNew.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import os
import numpy

if __name__ == '__main__':
    for bands in [30]:
        for session in range(1, 2):
            loadpath = 'D:/ProjectData/IEMOCAP-New-Again/Bands%d/' % bands
            netpath = 'D:/ProjectData/Determination/CTC/Data-01-Double-BLSTM/Bands-30-Session-0/'
            savepath = 'Bands-30-Session-%d/' % session
            # os.makedirs(savepath)

            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = LoaderTotal(
                loadpath=loadpath, session=session)

            for episode in range(50,100):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_Multi_BLSTM(trainData=trainData, trainLabel=trainScription,
                                                 trainSeqLength=trainSeq, featureShape=bands, numClass=3,
                                                 batchSize=32, startFlag=False, rnnLayers=1)
                    classifier.Load(loadpath=netpath + '%04d-Network' % episode)

                    numpy.save('TrainGroundLabel.npy', trainLabel)
                    numpy.save('TrainSeq.npy', trainSeq)
                    numpy.save('TestGroundLabel.npy', testLabel)
                    numpy.save('TestSeq.npy', testSeq)

                    inputData = trainData.copy()
                    inputSeq = trainSeq.copy()
                    inputLabel = trainLabel.copy()
                    totalLogits = classifier.LogitsOutput(testData=inputData, testSeq=inputSeq)

                    classifiedTestData, classifiedTestLabel = [], []
                    for indexX in range(len(inputData)):
                        for indexY in range(len(inputData[indexX])):
                            classifiedTestData.append(inputData[indexX][indexY])
                            if totalLogits[indexX][indexY] == 0:
                                classifiedTestLabel.append([1, 0, 0, 0, 0])
                            else:
                                classifiedTestLabel.append(numpy.concatenate(([0], inputLabel[indexX])))
                    print(numpy.shape(classifiedTestData), numpy.shape(classifiedTestLabel),
                          numpy.sum(classifiedTestLabel, axis=0))
                    numpy.save(savepath + 'TrainData-%d.npy' % episode, classifiedTestData)
                    numpy.save(savepath + 'TrainLabel-%d.npy' % episode, classifiedTestLabel)

                    inputData = testData.copy()
                    inputSeq = testSeq.copy()
                    inputLabel = testLabel.copy()
                    totalLogits = classifier.LogitsOutput(testData=inputData, testSeq=inputSeq)

                    classifiedTestData, classifiedTestLabel = [], []
                    for indexX in range(len(inputData)):
                        for indexY in range(len(inputData[indexX])):
                            classifiedTestData.append(inputData[indexX][indexY])
                            if totalLogits[indexX][indexY] == 0:
                                classifiedTestLabel.append([1, 0, 0, 0, 0])
                            else:
                                classifiedTestLabel.append(numpy.concatenate(([0], inputLabel[indexX])))
                    print(numpy.shape(classifiedTestData), numpy.shape(classifiedTestLabel),
                          numpy.sum(classifiedTestLabel, axis=0))
                    numpy.save(savepath + 'TestData-%d.npy' % episode, classifiedTestData)
                    numpy.save(savepath + 'TestLabel-%d.npy' % episode, classifiedTestLabel)
                    # exit()
