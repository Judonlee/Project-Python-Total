from CTC_Target.Loader.IEMOCAP_Loader import Load_MSP, Load_MSP_Part
import tensorflow
from CTC_Target.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
import os
import numpy

if __name__ == '__main__':
    bands = 30
    scope = 3
    for session in range(1, 4):
        for gender in ['F', 'M']:
            loadpath = 'E:/CTC_Target_MSP/Feature/Bands-%d/' % bands
            savepath = 'Result-CTC-LA-%d/Bands-%d/Session-%d-%s/' % (scope, bands, session, gender)
            netpath = 'E:/CTC_Target_MSP/CTC-MSP-LA-%d/Bands-%d-Session-%d-%s/%04d-Network'
            if os.path.exists(savepath): continue

            os.makedirs(savepath + 'Decode')
            os.makedirs(savepath + 'Logits')
            os.makedirs(savepath + 'SoftMax')

            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = Load_MSP_Part(
                loadpath=loadpath, appointSession=session, appointGender=gender)

            for episode in range(99, 100):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_LC_Attention(trainData=trainData, trainLabel=trainScription,
                                                  trainSeqLength=trainSeq, featureShape=len(trainData[0][0]),
                                                  numClass=5, rnnLayers=2, graphRevealFlag=False,
                                                  attentionScope=scope)
                    print('\nEpisode %d/100' % episode)
                    classifier.Load(loadpath=netpath % (scope, bands, session, gender, episode))
                    matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(
                        testData=testData, testLabel=testLabel, testSeq=testSeq)
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
