from MultiModalTest.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSession
import os
import tensorflow

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    for bands in [30]:
        loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d/' % bands
        for session in range(1):
            parameterpath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-Origin/Bands30/Session%d-LR/' % session
            savepath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-Origin-Result/Bands30/Session%d-LR/' % session
            if os.path.exists(savepath): continue

            if not os.path.exists(savepath): os.makedirs(savepath)
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSession(
                loadpath=loadpath, appointSession=session)
            for episode in range(100, 200):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_Multi_BLSTM(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                                 featureShape=bands, numClass=40, rnnLayers=2)
                    classifier.Load(loadpath=parameterpath + '%04d-Network' % episode)
                    loss = classifier.LossCalculation(testData=trainData, testLabel=trainLabel, testSeq=trainSeq)
                    print('\nEpisode %d Loss =%f' % (episode, loss))

                    with open(os.path.join(savepath, '%04d.csv' % episode), 'w') as file:
                        file.write(str(loss))
                    # exit()
