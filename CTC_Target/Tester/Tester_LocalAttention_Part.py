from CTC_Target.Loader.IEMOCAP_Loader import Load, Load_Part
import tensorflow
from CTC_Target.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
import os
import numpy

if __name__ == '__main__':
    part = 'MFCC'
    localAttentionScope = 5
    loadpath = 'E:/CTC_Target/Features/%s/' % part
    for session in range(1, 2):
        for gender in ['Female', 'Male']:
            savepath = 'Result-CTC-LA-%d-Part/%s-Session-%d-%s/' % (localAttentionScope, part, session, gender)
            netpath = 'E:/CTC_Target/CTC-LC-Attention-' + str(
                localAttentionScope) + '/%s-Session-%d-%s/%04d-Network'
            if os.path.exists(savepath): continue

            os.makedirs(savepath + 'Decode')
            os.makedirs(savepath + 'Logits')
            os.makedirs(savepath + 'SoftMax')

            trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load_Part(
                loadpath=loadpath, appointGender=gender, appointSession=session)

            for episode in range(100):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_LC_Attention(trainData=None, trainLabel=None, trainSeqLength=None,
                                                  featureShape=numpy.shape(testData[0])[1], numClass=5, rnnLayers=2,
                                                  graphRevealFlag=False, startFlag=False,
                                                  attentionScope=localAttentionScope)
                    print('\nEpisode %d/100' % episode)
                    classifier.Load(loadpath=netpath % (part, session, gender, episode))
                    matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                           testLabel=testlabel,
                                                                                           testSeq=testSeq)
                    print('\n\n')
                    print(matrixDecode)
                    print(matrixLogits)
                    print(matrixSoftMax)

                    with open(savepath + 'Decode/%04d.csv' % episode, 'w') as file:
                        for indexX in range(4):
                            for indexY in range(4):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixDecode[indexX][indexY]))
                            file.write('\n')
                    with open(savepath + 'Logits/%04d.csv' % episode, 'w') as file:
                        for indexX in range(4):
                            for indexY in range(4):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixLogits[indexX][indexY]))
                            file.write('\n')
                    with open(savepath + 'SoftMax/%04d.csv' % episode, 'w') as file:
                        for indexX in range(4):
                            for indexY in range(4):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixSoftMax[indexX][indexY]))
                            file.write('\n')

                    # exit()
