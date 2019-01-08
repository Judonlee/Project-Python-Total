from CTC_Target.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSession, LoaderLeaveOneSpeaker
import os
import tensorflow

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    for bands in [30, 40]:
        for attentionScope in [3, 5, 7]:
            loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d/' % bands
            for session in range(6):
                for gender in ['Female', 'Male']:
                    savepath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-LA=%d/Bands%d/Session%d-%s/' % (
                        attentionScope, bands, session, gender)
                    if os.path.exists(savepath): continue

                    if not os.path.exists(savepath): os.makedirs(savepath)
                    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSpeaker(
                        loadpath=loadpath, appointSession=session, appointGender=gender)
                    graph = tensorflow.Graph()
                    with graph.as_default():
                        classifier = CTC_LC_Attention(trainData=trainData, trainLabel=trainLabel,
                                                      trainSeqLength=trainSeq,
                                                      featureShape=30, numClass=40, rnnLayers=2,
                                                      attentionScope=attentionScope)
                        for episode in range(100):
                            print('\nEpisode %d : Total Loss = %f' % (episode, classifier.Train()))
                            classifier.Save(savepath=savepath + '%04d-Network' % episode)
