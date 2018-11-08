from CTC_Project_Again.Loader.IEMOCAP_Loader_New import LoaderTotal
import tensorflow
from CTC_Project_Again.ModelNew.BLSTM_CTC_CRF import BLSTM_CTC_CRF
import os
import numpy

if __name__ == '__main__':
    for gender in ['Female']:
        for bands in [30]:
            for session in range(1, 6):
                loadpath = 'D:/ProjectData/IEMOCAP-New-Again/Bands%d/' % bands
                netpath = 'D:/ProjectData/BrandNewCTC/Data-01-Single-BLSTM/Bands-%d-Session-%d/' % (
                    bands, session)

                trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = LoaderTotal(
                    loadpath=loadpath, session=session)
                testData = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
                testLabel = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
                testSeq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))

                for episode in range(99, 100):
                    # if os.path.exists(savepath + '%04d.csv' % episode): continue
                    if not os.path.exists(netpath + '%04d-Network.index' % episode): continue
                    graph = tensorflow.Graph()
                    with graph.as_default():
                        classifier = BLSTM_CTC_CRF(trainData=trainData, trainSeqLabel=trainScription,
                                                   trainGroundLabel=trainLabel, trainSeqLength=trainSeq,
                                                   featureShape=bands, numClass=4, rnnLayers=1, graphRevealFlag=True,
                                                   batchSize=32, startFlag=True)
                        for sample in tensorflow.global_variables():
                            print(sample)
                        print(len(tensorflow.global_variables()))
                        classifier.Load_CTC(loadpath=netpath + '%04d-Network' % episode)
                        print(classifier.information)
                        classifier.CRF_Train()
                        # logits = classifier.LogitsOutput(testData=testData, testSeq=testSeq)
                        exit()
                        # logits = classifier.LogitsOutput(testData=testData, testSeq=testSeq)
                        #
                        # file = open('Session-%d-%s.csv' % (session, gender), 'w')
                        # for indexX in range(len(logits)):
                        #     for indexY in range(testSeq[indexX]):
                        #         if indexY != 0: file.write(',')
                        #         file.write(str(logits[indexX][indexY]))
                        #     file.write('\n')
                        # file.close()
                        # exit()
