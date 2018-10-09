from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy
from CTC_Project_Again.Model.CTC_BLSTM import CTC_BLSTM
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
import os
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    numClass = 5
    for bands in [30, 40, 60, 80, 100, 120]:
        savepath = 'Records-CTC-Class%d-FAU/Bands-%d/' % (numClass, bands)
        if os.path.exists(savepath): continue
        os.makedirs(savepath)

        graph = tensorflow.Graph()
        with graph.as_default():
            [trainData, trainLabel, trainSeq] = numpy.load(
                'D:/ProjectData/FAU-AEC-Treated/IS2009-Class%d-Npy/Bands-%d/Ohm.npy' % (numClass, bands))
            trainScription = numpy.load(
                'D:/ProjectData/FAU-AEC-Treated/IS2009-Class%d-Transcription-Npy/Ohm.npy' % numClass)
            [testData, testLabel, testSeq] = numpy.load(
                'D:/ProjectData/FAU-AEC-Treated/IS2009-Class%d-Npy/Bands-%d/Mont.npy' % (numClass, bands))
            testScription = numpy.load(
                'D:/ProjectData/FAU-AEC-Treated/IS2009-Class%d-Transcription-Npy/Mont.npy' % numClass)

            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainScription,
                                                     trainSeq=trainSeq, testData=testData,
                                                     testLabel=testScription, testSeq=testSeq)
            print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(trainScription))
            print(numpy.shape(testData), numpy.shape(testLabel), numpy.shape(testSeq), numpy.shape(testScription))
            # exit()
            classifier = CTC_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                   featureShape=bands, numClass=numClass + 1, learningRate=1e-3, batchSize=64)
            print(classifier.information)

            for epoch in range(100):
                print('\rEpoch %d: Total Loss = %f' % (epoch, classifier.Train()))
                classifier.Save(savepath=savepath + '%04d-Network' % epoch)
