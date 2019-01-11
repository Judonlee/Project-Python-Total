from MultiModalTest.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSession
import os
import tensorflow

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    attentionScope = 3

    for bands in [30]:
        loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d/' % bands
        for session in range(1):
            parameterpath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-LA-%d/Bands30/Session%d-LR/' % (
                attentionScope, session)
            savepath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-LA-%d-Result/Bands30/Session%d-LR/' % (
                attentionScope, session)
            if os.path.exists(savepath): continue

            if not os.path.exists(savepath): os.makedirs(savepath)
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSession(
                loadpath=loadpath, appointSession=session)
            for episode in range(200):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_LC_Attention(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                                  featureShape=bands, numClass=40, rnnLayers=2,
                                                  attentionScope=attentionScope)
                    classifier.Load(loadpath=parameterpath + '%04d-Network' % episode)
                    loss = classifier.LossCalculation(testData=trainData, testLabel=trainLabel, testSeq=trainSeq)
                    print('\nEpisode %d Loss =%f' % (episode, loss))

                    with open(os.path.join(savepath, '%04d.csv' % episode), 'w') as file:
                        file.write(str(loss))
                    # exit()
