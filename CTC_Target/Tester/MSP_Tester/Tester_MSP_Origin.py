from CTC_Target.Loader.IEMOCAP_Loader import Load_MSP, Load_MSP_Part
import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import os
import numpy

if __name__ == '__main__':
    bands = 30

    for session in range(1, 2):
        for gender in ['F', 'M']:
            loadpath = 'E:/CTC_Target_MSP/Feature/Bands-%d/' % bands
            savepath = 'Result-CTC-Origin-MSP/Bands-%d/Session-%d-%s' % (bands, session, gender)
            netpath = 'E:/CTC_Target_MSP/CTC-MSP-Origin/Bands-%d-Session-%d/%04d-Network'
            if os.path.exists(savepath): continue

            os.makedirs(savepath + 'Decode')
            os.makedirs(savepath + 'Logits')
            os.makedirs(savepath + 'SoftMax')

            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = Load_MSP_Part(
                loadpath=loadpath, appointSession=session, appointGender=gender)

            for episode in range(100):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_Multi_BLSTM(trainData=None, trainLabel=None, trainSeqLength=None,
                                                 featureShape=bands, numClass=5, rnnLayers=2, graphRevealFlag=False,
                                                 startFlag=False)
                    print('\nEpisode %d/100' % episode)
                    classifier.Load(loadpath=netpath % (bands, session, episode))
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
