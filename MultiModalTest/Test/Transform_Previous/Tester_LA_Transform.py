import tensorflow
import os
from MultiModalTest.Model.Previous.CTC_LA_Transform import CTC_LA_Transform
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSpeaker

if __name__ == '__main__':
    punishment = 1
    for bands in [30, 40]:
        loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d-CNN/' % bands
        for attentionScope in [3, 5, 7]:
            for session in range(1, 6):
                for gender in ['Female', 'Male']:
                    parameterpath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-LA-%d/Bands%d/%04d-Network' % (
                        attentionScope, bands, 199)
                    calculatepath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-Transform-LA-%d-Punishment-%d/Bands%d/Session%d-%s/' % (
                        attentionScope, punishment, bands, session, gender)
                    savepath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-Transform-LA-%d-Punishment-%d-Result/Bands%d/Session%d-%s/' % (
                        attentionScope, punishment, bands, session, gender)
                    if not os.path.exists(calculatepath): continue
                    if os.path.exists(savepath): continue

                    if not os.path.exists(savepath): os.makedirs(savepath)
                    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSpeaker(
                        loadpath=loadpath, appointSession=session, appointGender=gender)

                    for episode in range(200):
                        graph = tensorflow.Graph()
                        with graph.as_default():
                            classifier = CTC_LA_Transform(trainData=trainData, trainLabel=trainLabel,
                                                          trainSeqLength=trainSeq, featureShape=bands, numClass=5,
                                                          rnnLayers=2, attentionScope=attentionScope,
                                                          graphRevealFlag=False, startFlag=False)
                            classifier.LoadPart(loadpath=parameterpath, alpha=punishment, flag='L1',
                                                graphRevealFlag=False)
                            print(calculatepath + '%04d-Network' % episode)
                            classifier.Load(loadpath=calculatepath + '%04d-Network' % episode)

                            matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(
                                testData=testData, testLabel=testLabel, testSeq=testSeq)

                            with open(savepath + '%04d-Decode.csv' % episode, 'w') as file:
                                for indexX in range(len(matrixDecode)):
                                    for indexY in range(len(matrixDecode[indexX])):
                                        if indexY != 0: file.write(',')
                                        file.write(str(matrixDecode[indexX][indexY]))
                                    file.write('\n')

                            with open(savepath + '%04d-Logits.csv' % episode, 'w') as file:
                                for indexX in range(len(matrixDecode)):
                                    for indexY in range(len(matrixDecode[indexX])):
                                        if indexY != 0: file.write(',')
                                        file.write(str(matrixLogits[indexX][indexY]))
                                    file.write('\n')

                            with open(savepath + '%04d-SoftMax.csv' % episode, 'w') as file:
                                for indexX in range(len(matrixDecode)):
                                    for indexY in range(len(matrixDecode[indexX])):
                                        if indexY != 0: file.write(',')
                                        file.write(str(matrixSoftMax[indexX][indexY]))
                                    file.write('\n')

                            # totalLoss, ctcLoss, punishmentLoss = classifier.LossCalculation(
                            #     testData=trainData, testLabel=trainLabel, testSeq=trainSeq)
                            #
                            # with open(savepath + '%04d-TrainLoss.csv' % episode, 'w') as file:
                            #     file.write(str(totalLoss[0]) + ',' + str(ctcLoss) + ',' + str(punishmentLoss[0]))
                            # exit()
                    # exit()
