import tensorflow
import os
from MultiModalTest.Model.Previous.CTC_LA_Transform import CTC_LA_Transform
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSpeaker
from MultiModalTest.TrainTemplate.TrainTimes200 import TrainTimes200

if __name__ == '__main__':
    punishment = 1
    for bands in [30, 40]:
        loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d-CNN/' % bands
        for attentionScope in [3, 5, 7]:
            for session in range(1, 6):
                for gender in ['Female', 'Male']:
                    parameterpath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-LA-%d/Bands%d/%04d-Network' % (
                        attentionScope, bands, 199)
                    savepath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-Transform-LA-%d-Punishment-%d/Bands%d/Session%d-%s/' % (
                        attentionScope, punishment, bands, session, gender)
                    if os.path.exists(savepath): continue

                    if not os.path.exists(savepath): os.makedirs(savepath)
                    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSpeaker(
                        loadpath=loadpath, appointSession=session, appointGender=gender)

                    graph = tensorflow.Graph()
                    with graph.as_default():
                        classifier = CTC_LA_Transform(trainData=trainData, trainLabel=trainLabel,
                                                      trainSeqLength=trainSeq, featureShape=bands, numClass=5,
                                                      rnnLayers=2, attentionScope=attentionScope, graphRevealFlag=False)
                        classifier.LoadPart(loadpath=parameterpath, alpha=punishment, flag='L1', graphRevealFlag=True)

                        TrainTimes200(classifier=classifier, savepath=savepath)

                    # exit()
