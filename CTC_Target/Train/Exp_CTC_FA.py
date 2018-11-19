from CTC_Target.Loader.IEMOCAP_Loader import Load, Load_Part
import tensorflow
from CTC_Target.Model.CTC_BLSTM_FA import CTC_Multi_FA
import os

if __name__ == '__main__':
    bands = 30
    loadpath = 'D:/ProjectData/CTC_Target/Features/Bands%d/' % bands
    for session in range(1, 6):
        for gender in ['Female', 'Male']:
            savepath = 'D:/ProjectData/CTC_Target/CTC-FA/Bands-%d-Session-%d-%s/' % (bands, session, gender)
            if os.path.exists(savepath): continue
            os.makedirs(savepath)
            trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_Part(
                loadpath=loadpath, appointSession=session, appointGender=gender)

            graph = tensorflow.Graph()
            with graph.as_default():
                classifier = CTC_Multi_FA(trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                                          featureShape=bands, numClass=5, rnnLayers=2, graphRevealFlag=False)
                print(classifier.information)
                for episode in range(100):
                    print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
                    classifier.Save(savepath=savepath + '%04d-Network' % episode)
                # exit()
