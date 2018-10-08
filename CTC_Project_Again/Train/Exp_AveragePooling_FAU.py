from CTC_Project_Again.Model.BLSTM_AveragePooling import BLSTM_AveragePooling
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
from CTC_Project_Again.Engine.TrainTestEngine import TrainTestEngine
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    classNumber = 2
    for bands in [30, 40, 60, 80, 100, 120]:
        savepath = 'D:/ProjectData/Project-CTC-Data/Records-AveragePooling-FAU-Class%d/Bands-' % classNumber + str(
            bands) + '/'
        graph = tensorflow.Graph()
        with graph.as_default():
            trainData, trainLabel, trainSeq = numpy.load(
                'D:/ProjectData/FAU-AEC-Treated/IS2009-Class%d-Npy/Bands-%d/Ohm.npy' % (classNumber, bands))
            testData, testLabel, testSeq = numpy.load(
                'D:/ProjectData/FAU-AEC-Treated/IS2009-Class%d-Npy/Bands-%d/Mont.npy' % (classNumber, bands))
            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                                                     testData=testData, testLabel=testLabel, testSeq=testSeq)
            classifier = BLSTM_AveragePooling(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                              featureShape=bands, numClass=classNumber, learningRate=1e-4,
                                              batchSize=64)
            print(classifier.information)
            TrainTestEngine(dataClass=dataClass, classifier=classifier, totalEpoch=100, savepath=savepath)
