from CTC_Project_Again.Loader.IEMOCAP_Loader_New import LoaderTotal
import tensorflow
from CTC_Project_Again.ModelNew.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import os
import numpy

if __name__ == '__main__':
    for gender in ['Male']:
        for bands in [30]:
            for session in range(1, 6):
                loadpath = 'D:/ProjectData/IEMOCAP-New-Again/Bands%d/' % bands
                netpath = 'D:/ProjectData/BrandNewCTC/Data-01-Single-BLSTM/Bands-%d-Session-%d/' % (
                    bands, session)
                savepath = 'D:/Data-Loss-01-Single-BLSTM-Again/Bands-%d-Session-%d-%s/' % (
                    bands, session, gender)
                if os.path.exists(savepath): continue
                os.makedirs(savepath)

                trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = LoaderTotal(
                    loadpath=loadpath, session=session)
                testData = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
                testLabel = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
                testSeq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))

                for episode in range(100):
                    if os.path.exists(savepath + '%04d.csv' % episode): continue
                    if not os.path.exists(netpath + '%04d-Network.index' % episode): continue
                    graph = tensorflow.Graph()
                    with graph.as_default():
                        classifier = CTC_Multi_BLSTM(trainData=trainData, trainLabel=trainScription,
                                                     trainSeqLength=trainSeq, featureShape=bands, numClass=3,
                                                     rnnLayers=1, graphRevealFlag=False, batchSize=32, startFlag=False)
                        classifier.Load(loadpath=netpath + '%04d-Network' % episode)
                        # exit()
                        loss = classifier.LossCalculation(testData=testData, testLabel=testScription, testSeq=testSeq)
                        print('\n', loss)

                        file = open(savepath + '%04d.csv' % episode, 'w')
                        file.write(str(loss))
                        file.close()
