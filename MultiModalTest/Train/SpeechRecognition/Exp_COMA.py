from MultiModalTest.Model.CTC_BLSTM_COMA import CTC_COMA_Attention
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSession, LoaderLeaveOneSpeaker
import os
import tensorflow
from MultiModalTest.TrainTemplate.TrainTimes200 import TrainTimes200

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    for bands in [30, 40]:
        loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d/' % bands
        for attentionScope in [3, 5, 7]:
            for session in range(1):
                savepath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-COMA-%d/Bands%d/Session%d/' % (
                    attentionScope, bands, session)
                if os.path.exists(savepath): continue

                if not os.path.exists(savepath): os.makedirs(savepath)
                trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSession(
                    loadpath=loadpath, appointSession=session)

                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_COMA_Attention(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                                    featureShape=bands, numClass=40, rnnLayers=2,
                                                    attentionScope=attentionScope, graphRevealFlag=True)
                    print(classifier.information)
                    # classifier.Train(learningRate=1E-3)
                    # exit()
                    TrainTimes200(classifier=classifier, savepath=savepath)
