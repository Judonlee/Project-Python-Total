from CTC_Project_Again.Model.BLSTM_FinalPooling import BLSTM_FinalPooling
import tensorflow
from __Base.DataClass import DataClass_TrainTest_Sequence
from CTC_Project_Again.Engine.TrainTestEngine import TrainTestEngine
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '3'

    classNumber = 2
    for bands in [30, 40, 60, 80, 100, 120]:
        savepath = 'D:/ProjectData/Project-CTC-Data/Records-FinalPooling-FAU-Class2/Bands-' + str(bands) + '/'
        graph = tensorflow.Graph()
        with graph.as_default():
            trainData, trainLabel, trainSeq = numpy.load(
                'D:/ProjectData/FAU-AEC-Treated/IS2009-Class%d-Npy/Bands-%d/Ohm.npy' % (classNumber, bands))
            testData, testLabel, testSeq = numpy.load(
                'D:/ProjectData/FAU-AEC-Treated/IS2009-Class%d-Npy/Bands-%d/Mont.npy' % (classNumber, bands))
            dataClass = DataClass_TrainTest_Sequence(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                                                     testData=testData, testLabel=testLabel, testSeq=testSeq)
            classifier = BLSTM_FinalPooling(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                            featureShape=bands, numClass=2, learningRate=1e-3, batchSize=64)
            print(classifier.information)
            # classifier.Train()
            # exit()
            TrainTestEngine(dataClass=dataClass, classifier=classifier, totalEpoch=100, savepath=savepath)
