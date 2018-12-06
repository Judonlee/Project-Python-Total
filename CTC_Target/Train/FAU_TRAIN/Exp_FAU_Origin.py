from CTC_Target.Loader.IEMOCAP_Loader import Load_FAU
import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import os
from CTC_Target.Train.FAU_TRAIN.LabelBalance import LabelBalance

if __name__ == '__main__':
    for bands in [30, 40]:
        loadpath = 'E:/CTC_Target_FAU/Features/Bands%d/' % bands
        savepath = 'CTC-Origin-FAU/Bands-%d/' % (bands)
        if os.path.exists(savepath): continue
        os.makedirs(savepath)
        trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_FAU(
            loadpath=loadpath)
        trainData, trainLabel, trainSeq, trainScription = LabelBalance(trainData=trainData, trainLabel=trainLabel,
                                                                       trainSeq=trainSeq, trainScription=trainScription)
        # exit()

        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_Multi_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                         featureShape=bands, numClass=6, rnnLayers=2, graphRevealFlag=False)
            print(classifier.information)
            for episode in range(100):
                print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
                classifier.Save(savepath=savepath + '%04d-Network' % episode)
            # exit()
