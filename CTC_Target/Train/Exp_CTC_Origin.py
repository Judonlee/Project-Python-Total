from CTC_Target.Loader.IEMOCAP_Loader import Load, LoadSpecialLabel
import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import os

if __name__ == '__main__':
    transcriptionPath = 'E:/CTC_Target/Features/PronouncingDictionaryDouble/'
    for part in ['Bands30', 'Bands40']:
        loadpath = 'E:/CTC_Target/Features/%s/' % part
        for session in range(1, 6):
            savepath = 'CTC-Origin_PDW/%s-Session-%d/' % (part, session)
            if os.path.exists(savepath): continue
            os.makedirs(savepath)
            trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = LoadSpecialLabel(
                loadpath=loadpath, appoint=session, transcriptionpath=transcriptionPath)

            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_Multi_BLSTM(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                             featureShape=len(trainData[0][0]), numClass=5, rnnLayers=2,
                                             graphRevealFlag=False)
                print(classifier.information)
                for episode in range(100):
                    print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
                    classifier.Save(savepath=savepath + '%04d-Network' % episode)
            # exit()
