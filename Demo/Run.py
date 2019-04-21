from Demo.Loader import Loader
from Demo.Model import NetworkStructure
import numpy

if __name__ == '__main__':
    trainData, trainLabel, valData, valLabel, testData, testLabel = Loader(
        appointSession=1, appointGender='F', datapath='D:/ProjectData/data/data/',
        labelpath='D:/ProjectData/data/one_hot_labels/')
    # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(valData), numpy.shape(valLabel),
    #       numpy.shape(testData), numpy.shape(testLabel))

    classifier = NetworkStructure(trainData=trainData, trainLabel=trainLabel, developData=valData,
                                  developLabel=valLabel, batchSize=32)
    classifier.Valid()
