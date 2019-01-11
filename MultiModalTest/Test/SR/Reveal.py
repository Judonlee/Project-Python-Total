from MultiModalTest.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSession
import os
import tensorflow
import matplotlib.pylab as plt
import librosa
import numpy

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    attentionScope = 3
    episode = 199

    bands = 30
    loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d/' % bands
    parameterpath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-LA-%d/Bands30/' % attentionScope

    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSession(
        loadpath=loadpath, appointSession=0)

    graph = tensorflow.Graph()
    with graph.as_default():
        classifier = CTC_LC_Attention(
            trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq, featureShape=bands,
            numClass=40, rnnLayers=2, attentionScope=attentionScope)
        classifier.Load(loadpath=parameterpath + '%04d-Network' % episode)

        result = classifier.session.run(
            fetches=classifier.parameters['Attention_Value'],
            feed_dict={classifier.dataInput: classifier.data[0:1],
                       classifier.seqLenInput: classifier.seqLen[0:1]})
        # print(result)
    plt.subplot(211)
    # result2 = numpy.copy(result)
    # for index in range(len(result)):
    #     result[index] -= result2[index - 1]
    plt.plot(numpy.arange(0, 0.01 * len(result) - 0.005, 0.01), result)
    plt.xlabel('Seconds (s)')
    plt.subplot(212)
    data, sr = librosa.load(
        r'D:\ProjectData\IEMOCAP\IEMOCAP-Voices\improve\Female\Session1\ang\Ses01F_impro01_F012.wav')
    print(data)
    print(numpy.shape(data))
    plt.plot(numpy.arange(0, (len(data)) / sr, 1 / sr), data)
    plt.xlabel('Seconds (s)')
    plt.show()
