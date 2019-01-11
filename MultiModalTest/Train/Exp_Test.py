from MultiModalTest.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
from MultiModalTest.Model.CTC_BLSTM_COMA import CTC_COMA_Attention
from MultiModalTest.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSession, LoaderLeaveOneSpeaker
import os
import tensorflow
from MultiModalTest.TrainTemplate.TrainTimes200 import TrainTimes200

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    bands = 30
    session = 0
    loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d/' % bands

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSession(
        loadpath=loadpath, appointSession=session)

    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = CTC_LC_Attention(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                      featureShape=bands, numClass=40, rnnLayers=2, attentionScope=3)
        for sample in tensorflow.global_variables()[8:10]:
            print(sample)
