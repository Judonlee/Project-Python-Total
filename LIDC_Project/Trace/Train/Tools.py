from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import numpy
from LIDC_Project.Loader.LIDC_Loader import LIDC_NewLoader


def Normalization_Treatment(trainData, testData):
    totalData = numpy.concatenate((trainData, testData), axis=0)
    totalData = scale(totalData)
    trainData = totalData[0:len(trainData)]
    testData = totalData[len(trainData):]
    return trainData, testData


def PCA_Treatment(trainData, testData, componentNumber):
    totalData = numpy.concatenate((trainData, testData), axis=0)
    pca = PCA(n_components=componentNumber)
    pca.fit(totalData)
    trainData = pca.transform(trainData)
    testData = pca.transform(testData)

    return Normalization_Treatment(trainData=trainData, testData=testData)


def LoaderPCA(leftPath, rightPath, part, pcaEpisode):
    trainDataLeft, trainLabelLeft, testDataLeft, testLabelLeft = LIDC_NewLoader(loadpath=leftPath, part=part)
    trainDataLeft = numpy.reshape(trainDataLeft,
                                  newshape=[-1, numpy.shape(trainDataLeft)[1] * numpy.shape(trainDataLeft)[2]])
    testDataLeft = numpy.reshape(testDataLeft,
                                 newshape=[-1, numpy.shape(testDataLeft)[1] * numpy.shape(testDataLeft)[2]])

    trainDataRight, trainLabelRight, testDataRight, testLabelRight = LIDC_NewLoader(loadpath=rightPath, part=part)
    trainDataRight = numpy.reshape(trainDataRight,
                                   newshape=[-1, numpy.shape(trainDataRight)[1] * numpy.shape(trainDataRight)[2]])
    testDataRight = numpy.reshape(testDataRight,
                                  newshape=[-1, numpy.shape(testDataRight)[1] * numpy.shape(testDataRight)[2]])

    # trainLabelLeft = numpy.argmax(trainLabelLeft, axis=1)
    # trainLabelRight = numpy.argmax(trainLabelRight, axis=1)
    # testLabelLeft = numpy.argmax(testLabelLeft, axis=1)
    # testLabelRight = numpy.argmax(testLabelRight, axis=1)
    #
    # for index in range(len(trainLabelLeft)):
    #     if trainLabelLeft[index] != trainLabelRight[index]:
    #         print('ERROR')
    #         exit()
    # for index in range(len(testLabelRight)):
    #     if testLabelLeft[index] != testLabelRight[index]:
    #         print('ERROR')
    #         exit()

    trainLabel = numpy.argmax(trainLabelLeft, axis=1)
    testLabel = numpy.argmax(testLabelLeft, axis=1)
    trainData = numpy.concatenate([trainDataLeft, trainDataRight], axis=1)
    testData = numpy.concatenate([testDataLeft, testDataRight], axis=1)

    trainData, testData = PCA_Treatment(trainData=trainData, testData=testData, componentNumber=pcaEpisode)
    print('PCA Remain %d - %d Pretreatment Completed' % (pcaEpisode, part), numpy.shape(trainData),
          numpy.shape(testData), numpy.shape(trainLabel), numpy.shape(testLabel))
    return trainData, trainLabel, testData, testLabel
