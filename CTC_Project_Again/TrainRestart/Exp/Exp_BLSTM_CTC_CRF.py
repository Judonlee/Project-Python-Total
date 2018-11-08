from CTC_Project_Again.Loader.IEMOCAP_Loader_New import LoaderTotal
import tensorflow
from CTC_Project_Again.ModelNew.BLSTM_CTC_CRF import BLSTM_CTC_CRF
import os

if __name__ == '__main__':
    part = 'Single'
    usedEpisode = 99
    for bands in [30]:
        for session in range(1, 6):
            for gender in ['Female', 'Male']:
                loadpath = 'D:/ProjectData/IEMOCAP-New-Again/Bands%d/' % bands
                netpath = 'D:/ProjectData/BrandNewCTC/Data-01-%s-BLSTM/Bands-%d-Session-%d/' % (
                    part, bands, session)
                savepath = 'Result-BLSTM-CTC-CRF-%s/Bands-%d-Session-%d/' % (part, bands, session)
                if os.path.exists(savepath): continue
                os.makedirs(savepath)

                trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = LoaderTotal(
                    loadpath=loadpath, session=session)

                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = BLSTM_CTC_CRF(trainData=trainData, trainSeqLabel=trainScription,
                                               trainGroundLabel=trainLabel, trainSeqLength=trainSeq,
                                               featureShape=bands, numClass=4, rnnLayers=1, graphRevealFlag=False,
                                               batchSize=32, startFlag=True)
                    classifier.Load_CTC(loadpath=netpath + '%04d-Network' % usedEpisode)
                    print(classifier.information)

                    for epoch in range(100):
                        print('\nEpoch %d: Total Loss = %f' % (epoch, classifier.CRF_Train()))
                        classifier.Save(savepath=savepath + '%04d-Network' % epoch)
