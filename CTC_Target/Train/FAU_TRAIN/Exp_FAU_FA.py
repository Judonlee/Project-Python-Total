from CTC_Target.Loader.IEMOCAP_Loader import Load_FAU
import tensorflow
from CTC_Target.Model.CTC_BLSTM_FA import CTC_Multi_FA
import os

if __name__ == '__main__':
    for bands in [30, 40]:
        loadpath = 'D:/ProjectData/FAU-AEC-Treated/Features/Bands%d/' % bands
        savepath = 'CTC-Origin-FA/Bands-%d/' % (bands)
        if os.path.exists(savepath): continue
        os.makedirs(savepath)
        trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_FAU(
            loadpath=loadpath)

        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_Multi_FA(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                      featureShape=bands, numClass=6, rnnLayers=2, graphRevealFlag=False)
            print(classifier.information)
            for episode in range(100):
                print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
                classifier.Save(savepath=savepath + '%04d-Network' % episode)
            # exit()
