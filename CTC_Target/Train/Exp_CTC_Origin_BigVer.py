from CTC_Target.Loader.IEMOCAP_Loader import Load, Load_Part, LoadSpecialLabel
import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM_BigVer import CTC_Multi_BLSTM_BigVer
import os

if __name__ == '__main__':
    rnnLayers = 4
    fullyConnectedLayers = 4
    for part in ['Bands30', 'Bands40']:
        loadpath = 'E:/CTC_Target/Features/%s/' % part
        for session in range(0, 6):
            for gender in ['F', 'M']:
                savepath = 'E:/CTC-Transform/IEMOCAP-Total-RN=%d-FC=%d/%s-Session-%d-%s/' % (
                    rnnLayers, fullyConnectedLayers, part, session, gender)

                if os.path.exists(savepath): continue
                os.makedirs(savepath)
                trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_Part(
                    loadpath=loadpath, appointGender=gender, appointSession=session)

                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_Multi_BLSTM_BigVer(
                        trainData=trainData, trainLabel=trainScription, trainSeqLength=trainSeq,
                        featureShape=len(trainData[0][0]), numClass=5, rnnLayers=rnnLayers,
                        fullyConnectedLayers=fullyConnectedLayers, graphRevealFlag=False)
                    print(classifier.information)
                    # exit()
                    for episode in range(100):
                        print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
                        classifier.Save(savepath=savepath + '%04d-Network' % episode)
                # exit()
