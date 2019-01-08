from MultiModalTest.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSession
import os
import tensorflow

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    for bands in [30, 40]:
        loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d/' % bands
        for session in range(6):
            savepath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-Origin/Bands30/Session%d/' % session
            if os.path.exists(savepath): continue

            if not os.path.exists(savepath): os.makedirs(savepath)
            trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSession(
                loadpath=loadpath, appointSession=session)
            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_Multi_BLSTM(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                             featureShape=30, numClass=40, rnnLayers=2)
                for episode in range(100):
                    print('\nEpisode %d : Total Loss = %f' % (episode, classifier.Train()))
                    classifier.Save(savepath=savepath + '%04d-Network' % episode)
