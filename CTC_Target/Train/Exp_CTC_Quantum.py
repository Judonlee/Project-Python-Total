from CTC_Target.Loader.IEMOCAP_Loader import Load, Load_Part
import tensorflow
from CTC_Target.Model.CTC_BLSTM_QuantumAttention import CTC_QuantumAttention
import os
import numpy

if __name__ == '__main__':
    for part in ['Bands30', 'Bands40']:
        loadpath = 'E:/CTC_Target/Features/%s/' % part
        for session in range(1, 6):
            for gender in ['Female', 'Male']:
                savepath = 'CTC-Quantum/%s-Session-%d-%s/' % (part, session, gender)
                if os.path.exists(savepath): continue
                os.makedirs(savepath)
                trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_Part(
                    loadpath=loadpath, appointSession=session, appointGender=gender)

                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_QuantumAttention(trainData=trainData, trainLabel=trainScription,
                                                      trainSeqLength=trainSeq,
                                                      featureShape=numpy.shape(trainData[0])[1], numClass=5,
                                                      rnnLayers=2, graphRevealFlag=False)
                    print(classifier.information)
                    for episode in range(100):
                        print('\nEpisode %d/100 : Total Loss = %f\n' % (episode, classifier.Train()), end='')
                        classifier.Save(savepath=savepath + '%04d-Network' % episode)
                    # exit()
