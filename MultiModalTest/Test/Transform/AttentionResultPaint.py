from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSpeaker
from MultiModalTest.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
from MultiModalTest.Model.Transform.CTC_LA_Transform import CTC_LA_Transform
import tensorflow
import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands30-Seq/'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSpeaker(
        loadpath=loadpath, appointSession=1, appointGender='Female')

    parameterPath = 'E:/ProjectData_SpeechRecognition/Transform/IEMOCAP-Tran-LA-3-Punishment-%d/Session1-Female/%04d-Network'
    for punishment in [1, 10, 100, 1000, 10000]:
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_LA_Transform(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                          featureShape=30, emotionClass=5, phonemeClass=40, rnnLayers=2,
                                          attentionScope=3, punishmentDegree=punishment, startFlag=False,
                                          initialParameterPath=parameterPath % (punishment, 99))
            plt.plot(classifier.session.run(fetches=classifier.parameters['Attention_Value'],
                                            feed_dict={classifier.dataInput: testData[0:1],
                                                       classifier.seqLenInput: testSeq[0:1],
                                                       classifier.punishmentInput: punishment}),
                     label='Origin Attention')
            plt.plot(classifier.session.run(fetches=classifier.parameters['Emotion_Attention_Value'],
                                            feed_dict={classifier.dataInput: testData[0:1],
                                                       classifier.seqLenInput: testSeq[0:1],
                                                       classifier.punishmentInput: punishment}),
                     label='Punishment=%d' % punishment)
            plt.legend()
            plt.title('Punishment = %d' % punishment)
            plt.xlabel('Train Episode')
            plt.ylabel('Loss')
            plt.show()
