from CTC_Target.Loader.IEMOCAP_Loader import Load_FAU
import tensorflow
from CTC_Target.Model.CTC_BLSTM_COMA import CTC_COMA_Attention
import os
import numpy

if __name__ == '__main__':
    for bands in [30, 40]:
        for attentionScope in [3, 5, 7]:
            loadpath = 'D:/ProjectData/FAU-AEC-Treated/Features/Bands%d/' % bands
            savepath = 'Result-CTC-COMA-%d/Bands-%d/' % (attentionScope, bands)
            netpath = 'E:/CTC_Target_FAU/CTC-COMA-%d/Bands-%d/%04d-Network'
            if os.path.exists(savepath): exit()

            os.makedirs(savepath + 'Decode')
            os.makedirs(savepath + 'Logits')
            os.makedirs(savepath + 'SoftMax')

            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = Load_FAU(
                loadpath=loadpath)

            for episode in range(100):
                if os.path.exists(savepath + 'Decode/%04d.csv' % episode): continue
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_COMA_Attention(trainData=None, trainLabel=None, trainSeqLength=None,
                                                    featureShape=bands, numClass=6, rnnLayers=2, graphRevealFlag=False,
                                                    startFlag=False, attentionScope=attentionScope)
                    print('\nEpisode %d/100' % episode)
                    classifier.Load(loadpath=netpath % (attentionScope, bands, episode))
                    matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethodsWithLen(
                        testData=testData, testLabel=testLabel, testSeq=testSeq)
                    print('\n\n')
                    print(matrixDecode)
                    print(matrixLogits)
                    print(matrixSoftMax)

                    with open(savepath + 'Decode/%04d.csv' % episode, 'w') as file:
                        for indexX in range(5):
                            for indexY in range(5):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixDecode[indexX][indexY]))
                            file.write('\n')
                    with open(savepath + 'Logits/%04d.csv' % episode, 'w') as file:
                        for indexX in range(5):
                            for indexY in range(5):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixLogits[indexX][indexY]))
                            file.write('\n')
                    with open(savepath + 'SoftMax/%04d.csv' % episode, 'w') as file:
                        for indexX in range(5):
                            for indexY in range(5):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixSoftMax[indexX][indexY]))
                            file.write('\n')

    # exit()
