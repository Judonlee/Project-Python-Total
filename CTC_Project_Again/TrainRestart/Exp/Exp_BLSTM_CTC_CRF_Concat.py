from CTC_Project_Again.Loader.IEMOCAP_Loader_New import LoaderTotal
import tensorflow
from CTC_Project_Again.ModelNew.BLSTM_CTC_CRF_BLSTM_Concat import BLSTM_CTC_CRF
import os

if __name__ == '__main__':
    for bands in [30]:
        for session in range(1, 6):
            for gender in ['Female', 'Male']:
                loadpath = 'D:/ProjectData/IEMOCAP-New-Again/Bands%d/' % bands
                netpath = 'D:/ProjectData/Determination/CTC/Data-01-Double-BLSTM/Bands-30-Session-0/0099-Network'
                savepath = 'CRF-Concat/Bands-%d-Session-%d/' % (bands, session)
                if os.path.exists(savepath): continue
                os.makedirs(savepath)

                trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = LoaderTotal(
                    loadpath=loadpath, session=session)

                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = BLSTM_CTC_CRF(trainData=testData, trainSeqLabel=testScription,
                                               trainGroundLabel=testLabel, trainSeqLength=testSeq,
                                               featureShape=bands, numClass=4, rnnLayers=2, graphRevealFlag=False,
                                               batchSize=32, startFlag=True)
                    classifier.Load_CTC(loadpath=netpath)
                    print(classifier.information)

                    for epoch in range(100):
                        print('\nEpoch %d: Total Loss = %f' % (epoch, classifier.CRF_Train()))
                        classifier.Save(savepath=savepath + '%04d-Network' % epoch)
