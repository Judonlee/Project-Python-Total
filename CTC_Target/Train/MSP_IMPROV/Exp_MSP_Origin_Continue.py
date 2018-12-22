from CTC_Target.Loader.IEMOCAP_Loader import Load_MSP
import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import os

if __name__ == '__main__':
    part = 'Bands-30'
    session = 5
    startPosition = 78
    loadpath = 'E:/CTC_Target_MSP/Feature/%s/' % part
    savepath = 'E:/CTC_Target_MSP/CTC-MSP-Origin/%s-Session-%d/' % (part, session)

    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_MSP(
        loadpath=loadpath, appointSession=session)

    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = CTC_Multi_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                     featureShape=len(trainData[0][0]), numClass=5, rnnLayers=2,
                                     graphRevealFlag=False)
        print(classifier.information)
        classifier.Load(loadpath=savepath + '%04d-Network' % startPosition)
        for episode in range(startPosition + 1, 100):
            print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
            classifier.Save(savepath=savepath + '%04d-Network' % episode)
    # exit()
