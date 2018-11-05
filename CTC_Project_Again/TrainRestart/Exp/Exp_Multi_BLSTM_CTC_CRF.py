import tensorflow
from CTC_Project_Again.ModelNew.Multi_BLSTM_CTC_CRF import BLSTM_CTC_CRF
import os
import numpy
from CTC_Project_Again.Loader.IEMOCAP_Loader_New import Loader

if __name__ == '__main__':
    used = 'UA'
    for bands in [30]:
        for session in range(1, 6):
            for gender in ['Female', 'Male']:
                loadpath = 'D:/ProjectData/IEMOCAP-New/Bands%d/' % bands
                netpath = 'D:/ProjectData/BrandNewCTC/DoubleBLSTM-Choosed/Bands-%d-Session-%d-%s/' % (
                    bands, session, gender)
                savepath = 'Double_BLSTM_CTC_CRF_%s/Bands-%d-Session-%d-%s/' % (used, bands, session, gender)

                if os.path.exists(savepath): continue
                os.makedirs(savepath)

                trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(loadpath=loadpath,
                                                                                       session=session)
                testData = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
                testLabel = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
                testSeq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))

                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = BLSTM_CTC_CRF(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                               featureShape=bands, numClass=5, rnnLayers=2, graphRevealFlag=False,
                                               batchSize=32, startFlag=True)
                    classifier.Load_CTC(loadpath=netpath + used)
                    for episode in range(100):
                        if episode < 5:
                            matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(
                                testData=testData, testLabel=testLabel, testSeq=testSeq)
                            print('Session %d Gender %s' % (session, gender))
                            print(matrixDecode)
                        print('\nEpisode %d Total Loss = %f\n' % (episode, classifier.CRF_Train()))
                        classifier.Save(savepath=savepath + '%04d-Network' % episode)
