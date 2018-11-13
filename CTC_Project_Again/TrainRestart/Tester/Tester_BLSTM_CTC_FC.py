from CTC_Project_Again.Loader.IEMOCAP_Loader_New import LoaderTotal
import tensorflow
from CTC_Project_Again.ModelNew.BLSTM_CTC_FC_Concat import BLSTM_CTC_FC
import os

if __name__ == '__main__':
    for bands in [30]:
        for session in range(1, 2):
            loadpath = 'D:/ProjectData/IEMOCAP-New-Again/Bands%d/' % bands
            savepath = 'Result-Train/Bands-%d-Session-%d/' % (bands, session)
            os.makedirs(savepath)

            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = LoaderTotal(
                loadpath=loadpath, session=session)

            for episode in range(90):
                netpath = 'D:/ProjectData/Determination/20181112-Result/CRF-FC/Bands-%d-Session-%d/%04d-Network' % (
                    bands, session, episode)
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = BLSTM_CTC_FC(trainData=trainData, trainSeqLabel=trainScription,
                                              trainGroundLabel=trainLabel, trainSeqLength=trainSeq,
                                              featureShape=bands, numClass=4, rnnLayers=2, graphRevealFlag=False,
                                              batchSize=32, startFlag=False, learningRate=1e-4)
                    classifier.Load(loadpath=netpath)
                    matrix = classifier.FC_Test(testData=trainData, testLabel=trainLabel, testSeq=trainSeq)

                    file = open(savepath + '%04d.csv' % episode, 'w')
                    for indexX in range(4):
                        for indexY in range(4):
                            if indexY != 0: file.write(',')
                            file.write(str(matrix[indexX][indexY]))
                        file.write('\n')
                    file.close()
                    # exit()
