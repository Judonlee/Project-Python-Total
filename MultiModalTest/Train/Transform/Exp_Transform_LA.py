from MultiModalTest.Model.Transform.CTC_LA_Transform import CTC_LA_Transform
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSpeaker
from MultiModalTest.TrainTemplate.TrainTimes200 import TrainTimes200_Emotion
import os
import tensorflow

if __name__ == '__main__':
    punishment = 1

    for bands in [30]:
        for attentionScope in [3, 5, 7]:
            for session in range(1, 6):
                for gender in ['Female', 'Male']:
                    loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d-Seq/' % (bands)
                    parameterpath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-LA-%d/Bands%d/%04d-Network' % (
                        attentionScope, bands, 199)

                    savepath = 'E:/ProjectData_SpeechRecognition/Transform/IEMOCAP-Tran-LA-%d-Punishment-%d/Session%d-%s/' % (
                        attentionScope, punishment, session, gender)
                    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSpeaker(
                        loadpath=loadpath, appointSession=session, appointGender=gender)

                    if os.path.exists(savepath): continue
                    graph = tensorflow.Graph()
                    with graph.as_default():
                        classifier = CTC_LA_Transform(trainData=trainData, trainLabel=trainLabel,
                                                      trainSeqLength=trainSeq, featureShape=bands, emotionClass=5,
                                                      phonemeClass=40, rnnLayers=2, attentionScope=attentionScope,
                                                      punishmentDegree=punishment, startFlag=True,
                                                      initialParameterPath=parameterpath, graphRevealFlag=True)
                        TrainTimes200_Emotion(classifier=classifier, savepath=savepath)
                        # exit()
