from CTC_Target.Loader.IEMOCAP_Loader import Load
import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import os
import numpy

if __name__ == '__main__':
    bands = 30
    loadpath = 'D:/ProjectData/CTC_Target/Features/Bands%d/' % bands
    for session in range(1, 6):
        for gender in ['Female', 'Male']:
            savepath = 'Result-CTC-Origin/Bands-%d-Session-%d/' % (bands, session)
            netpath = 'D:/ProjectData/CTC_Target/CTC-Origin/Bands-%d-Session-%d/%04d-Network'
            if os.path.exists(savepath): continue

            os.makedirs(savepath + 'Decode')
            os.makedirs(savepath + 'Logits')
            os.makedirs(savepath + 'SoftMax')

            # trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load(
            #     loadpath=loadpath, appoint=session)
            testData = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
            testLabel = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
            testSeq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))

            for episode in range(100):
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = CTC_Multi_BLSTM(trainData=None, trainLabel=None, trainSeqLength=None,
                                                 featureShape=bands, numClass=5, rnnLayers=2, graphRevealFlag=False,
                                                 startFlag=False)
                    print('\nEpisode %d/100' % episode)
                    classifier.Load(loadpath=netpath % (bands, session, episode))
                    matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                           testLabel=testLabel,
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
