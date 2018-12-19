from CTC_Target.Loader.IEMOCAP_Loader import Load_MSP_Part
import tensorflow
from CTC_Target.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
import os

if __name__ == '__main__':
    for part in ['Bands-30', 'Bands-40']:
        loadpath = 'D:/ProjectData/MSP-IMPROVE/Feature/%s/' % part
        for scope in [3, 5, 7]:
            for session in range(1, 6):
                for gender in ['F', 'M']:
                    savepath = 'CTC-MSP-LA-%d/%s-Session-%d-%s/' % (scope, part, session, gender)
                    if os.path.exists(savepath): continue
                    os.makedirs(savepath)
                    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_MSP_Part(
                        loadpath=loadpath, appointSession=session, appointGender=gender)

                    graph = tensorflow.Graph()
                    with graph.as_default():
                        classifier = CTC_LC_Attention(trainData=trainData, trainLabel=trainScription,
                                                      trainSeqLength=trainSeq, featureShape=len(trainData[0][0]),
                                                      numClass=5, rnnLayers=2, graphRevealFlag=False,
                                                      attentionScope=scope)
                        print(classifier.information)
                        for episode in range(100):
                            print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
                            classifier.Save(savepath=savepath + '%04d-Network' % episode)
                    # exit()
