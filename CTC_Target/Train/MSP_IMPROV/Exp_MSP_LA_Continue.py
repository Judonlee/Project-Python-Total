from CTC_Target.Loader.IEMOCAP_Loader import Load_MSP_Part
import tensorflow
from CTC_Target.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
import os

if __name__ == '__main__':
    part = 'Bands-30'
    scope = 3
    session = 2
    gender = 'M'
    startPosition = 97

    loadpath = 'E:/CTC_Target_MSP/Feature/%s/' % part

    savepath = 'E:/CTC_Target_MSP/CTC-MSP-LA-%d/%s-Session-%d-%s/' % (scope, part, session, gender)

    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_MSP_Part(
        loadpath=loadpath, appointSession=session, appointGender=gender)

    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = CTC_LC_Attention(trainData=trainData, trainLabel=trainScription,
                                      trainSeqLength=trainSeq, featureShape=len(trainData[0][0]),
                                      numClass=5, rnnLayers=2, graphRevealFlag=False,
                                      attentionScope=scope, startFlag=False)
        print(classifier.information)
        classifier.Load(loadpath=savepath + '%04d-Network' % startPosition)
        for episode in range(startPosition + 1, 100):
            print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
            classifier.Save(savepath=savepath + '%04d-Network' % episode)
    # exit()
