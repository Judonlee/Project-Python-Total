from MultiModalTest.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSpeaker
from MultiModalTest.TrainTemplate.TrainTimes200 import TrainTimes200_Emotion
import os
import tensorflow

if __name__ == '__main__':
    punishment = 1

    for bands in [30]:
        for attentionScope in [3]:
            for session in range(1, 2):
                for gender in ['Female']:
                    loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d/' % (bands)
                    parameterpath = r'E:\ProjectData_SpeechRecognition\Transform\IEMOCAP-Tran-LA-3-Punishment-1\Session1-Female\0000-Network'

                    # savepath = 'E:/ProjectData_SpeechRecognition/Transform/IEMOCAP-Tran-LA-%d-Punishment-%d/Session%d-%s/' % (
                    #     attentionScope, punishment, session, gender)
                    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSpeaker(
                        loadpath=loadpath, appointSession=0, appointGender=gender)

                    # if os.path.exists(savepath): continue
                    graph = tensorflow.Graph()
                    with graph.as_default():
                        classifier = CTC_LC_Attention(trainData=trainData, trainLabel=trainLabel,
                                                      trainSeqLength=trainSeq, featureShape=bands, rnnLayers=2,
                                                      attentionScope=attentionScope, startFlag=False,
                                                      graphRevealFlag=True, numClass=40)
                        classifier.Load(parameterpath)
                        print(classifier.LossCalculation(trainData, trainLabel, trainSeq))
                        # TrainTimes200_Emotion(classifier=classifier, savepath=savepath)
                        # exit()
