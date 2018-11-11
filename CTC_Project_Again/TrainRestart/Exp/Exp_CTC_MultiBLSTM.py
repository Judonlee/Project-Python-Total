from CTC_Project_Again.Loader.IEMOCAP_Loader_New import LoaderTotal
from CTC_Project_Again.ModelNew.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import tensorflow
import os

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    for bands in [30, 40, 60, 80, 100, 120]:
        loadpath = 'D:/ProjectData/IEMOCAP-New-Again/Bands%d/' % bands
        savepath = 'Data-01-Single-BLSTM/Bands-%d-Session-%d/' % (bands, 0)
        if os.path.exists(savepath): continue
        os.makedirs(savepath)
        trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = LoaderTotal(
            loadpath=loadpath, session=0)

        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_Multi_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                         featureShape=bands, numClass=3, rnnLayers=1, graphRevealFlag=False,
                                         batchSize=32)
            print(classifier.information)
            # exit()
            for epoch in range(100):
                print('\nEpoch %d: Total Loss = %f' % (epoch, classifier.Train()))
                classifier.Save(savepath=savepath + '%04d-Network' % epoch)
