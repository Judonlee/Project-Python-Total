from MultiModalTest.Model.Transform.CTC_LA_Transform import CTC_LA_Transform
from MultiModalTest.Loader.IEMOCAP_Loader import LoaderLeaveOneSpeaker
import os
import tensorflow

if __name__ == '__main__':
    punishment = 1

    for attentionScope in [3, 5, 7]:
        for bands in [30, 40]:
            for session in range(1, 6):
                for gender in ['Female', 'Male']:
                    print('Treating Attention Scope = %d Bands = %d Part = Session %d - %s' % (
                        attentionScope, bands, session, gender))

                    loadpath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands%d-Seq/' % (bands)
                    parameterpath = 'E:/ProjectData_SpeechRecognition/Transform/IEMOCAP-Tran-LA-%d-Punishment-%d/Session%d-%s/' % (
                        attentionScope, punishment, session, gender)
                    if not os.path.exists(parameterpath): continue

                    savepath = 'E:/ProjectData_SpeechRecognition/Transform/IEMOCAP-Tran-LA-%d-Punishment-%d-Result/Session%d-%s/' % (
                        attentionScope, punishment, session, gender)
                    if os.path.exists(savepath): continue
                    os.makedirs(savepath)

                    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = LoaderLeaveOneSpeaker(
                        loadpath=loadpath, appointSession=session, appointGender=gender)

                    for episode in range(99, 100):
                        graph = tensorflow.Graph()
                        with graph.as_default():
                            classifier = CTC_LA_Transform(
                                trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq, featureShape=bands,
                                emotionClass=5, phonemeClass=40, rnnLayers=2, attentionScope=attentionScope,
                                punishmentDegree=punishment, startFlag=False, graphRevealFlag=False,
                                initialParameterPath=parameterpath + '%04d-Network' % episode)
                            # classifier.Load(loadpath=parameterpath)
                            totalLoss, totalCTCLoss, totalPunishmentLoss, matrixDecode, matrixLogits, matrixSoftMax = \
                                classifier.TestEmotion(testData=trainData, testLabel=trainLabel, testSeq=trainSeq)

                            with open(savepath + 'TestLoss-%04d.csv' % episode, 'w') as file:
                                file.write(str(totalLoss) + ',' + str(totalCTCLoss) + ',' + str(totalPunishmentLoss))

                            with open(savepath + 'Decode-%04d.csv' % episode, 'w') as file:
                                for indexX in range(4):
                                    for indexY in range(4):
                                        if indexY != 0: file.write(',')
                                        file.write(str(matrixDecode[indexX][indexY]))
                                    file.write('\n')

                            with open(savepath + 'Logits-%04d.csv' % episode, 'w') as file:
                                for indexX in range(4):
                                    for indexY in range(4):
                                        if indexY != 0: file.write(',')
                                        file.write(str(matrixLogits[indexX][indexY]))
                                    file.write('\n')

                            with open(savepath + 'SoftMax-%04d.csv' % episode, 'w') as file:
                                for indexX in range(4):
                                    for indexY in range(4):
                                        if indexY != 0: file.write(',')
                                        file.write(str(matrixSoftMax[indexX][indexY]))
                                    file.write('\n')
