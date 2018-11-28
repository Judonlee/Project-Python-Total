from CTC_Target.Loader.IEMOCAP_Loader import Load_FAU
import tensorflow
from CTC_Target.Model.CTC_BLSTM_FA import CTC_Multi_FA
import os
import numpy

if __name__ == '__main__':
    bands = 30
    loadpath = 'D:/ProjectData/FAU-AEC-Treated/Features/Bands%d/' % bands
    savepath = 'Result-CTC-Origin/Bands-%d/' % bands
    netpath = 'E:/CTC_Target_FAU/CTC-Origin-FAU/Bands-%d/%04d-Network'
    if os.path.exists(savepath): exit()

    os.makedirs(savepath + 'Decode')
    os.makedirs(savepath + 'Logits')
    os.makedirs(savepath + 'SoftMax')

    trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = Load_FAU(
        loadpath=loadpath)

    for episode in range(100):
        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = CTC_Multi_FA(trainData=None, trainLabel=None, trainSeqLength=None,
                                      featureShape=bands, numClass=6, rnnLayers=2, graphRevealFlag=False,
                                      startFlag=False)
            print('\nEpisode %d/100' % episode)
            classifier.Load(loadpath=netpath % (bands, episode))
            matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                   testLabel=testLabel,
                                                                                   testSeq=testSeq)
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
