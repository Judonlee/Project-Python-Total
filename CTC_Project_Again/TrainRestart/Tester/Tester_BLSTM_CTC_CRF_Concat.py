import tensorflow
import numpy
from CTC_Project_Again.Loader.IEMOCAP_Loader_New import LoaderTotal
from CTC_Project_Again.ModelNew.BLSTM_CTC_CRF_BLSTM_Concat import BLSTM_CTC_CRF
import os

if __name__ == '__main__':

    for bands in [30]:
        for session in range(1, 2):
            loadpath = 'D:/ProjectData/IEMOCAP-New/Bands%d/' % bands
            netpath = 'D:/ProjectData/Determination/CRF-Concat/Bands-%d-Session-%d/' % (bands, session)

            savepath = 'D:/ProjectData/Determination/Result-CRF-Concat/Bands-%d-Session-%d/' % (
                bands, session)

            # if os.path.exists(savepath): continue
            # os.makedirs(savepath)

            trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = LoaderTotal(
                loadpath=loadpath, session=session)

            # testData = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
            # testLabel = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
            # testSeq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))

            for episode in range(43, 100):
                if not os.path.exists(netpath + '%04d-Network.meta' % episode): continue
                graph = tensorflow.Graph()
                with graph.as_default():
                    classifier = BLSTM_CTC_CRF(trainData=testData, trainSeqLabel=testScription,
                                               trainGroundLabel=testLabel, trainSeqLength=testSeq,
                                               featureShape=bands, numClass=4, rnnLayers=2, graphRevealFlag=False,
                                               batchSize=32, startFlag=False)
                    classifier.Load(loadpath=netpath + '%04d-Network' % episode)
                    # classifier.CRF_Train()
                    # exit()
                    matrix = classifier.Test_CRF(testData=testData, testLabel=testLabel, testSeq=testSeq)

                    file = open(savepath + '%04d.csv' % episode, 'w')
                    for indexX in range(4):
                        for indexY in range(4):
                            if indexY != 0: file.write(',')
                            file.write(str(matrix[indexX][indexY]))
                        file.write('\n')
                    file.close()
