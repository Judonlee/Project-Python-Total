from CTC_Project_Again.Loader.IEMOCAP_Loader_New import Loader
import tensorflow
from CTC_Project_Again.ModelNew.Multi_BLSTM_CTC_CRF import BLSTM_CTC_CRF
import os
import numpy

if __name__ == '__main__':
    used = 'UA'
    for bands in [30]:
        for session in range(1, 6):
            for gender in ['Female', 'Male']:
                loadpath = 'D:/ProjectData/IEMOCAP-New/Bands%d/' % bands
                netpath = 'D:/ProjectData/BrandNewCTC/DoubleBLSTM-Choosed/Bands-%d-Session-%d-%s/%s' % (
                    bands, session, gender, used)

                trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(loadpath=loadpath,
                                                                                       session=session)
                testData = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
                testLabel = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
                testSeq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))

                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = BLSTM_CTC_CRF(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                               featureShape=bands, numClass=5, rnnLayers=2, graphRevealFlag=True,
                                               batchSize=32, startFlag=True)
                    # for sample in tensorflow.global_variables():
                    #     print(sample)
                    # print(len(tensorflow.global_variables()))
                    classifier.Load_CTC(loadpath=netpath)
                    matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                           testLabel=testLabel,
                                                                                           testSeq=testSeq)
                    print(matrixDecode)
                    classifier.CRF_Train()
                    matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                           testLabel=testLabel,
                                                                                           testSeq=testSeq)
                    print(matrixDecode)
                    exit()
