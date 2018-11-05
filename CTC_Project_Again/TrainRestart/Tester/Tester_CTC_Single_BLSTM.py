from CTC_Project_Again.Loader.IEMOCAP_Loader_New import Loader
import tensorflow
from CTC_Project_Again.ModelNew.CTC_Single_Origin import CTC_BLSTM
import os
import numpy

if __name__ == '__main__':
    for gender in ['Male']:
        for bands in [30]:
            for session in range(1, 6):
                loadpath = 'D:/ProjectData/Project-CTC-Data/Csv-Npy/Bands%d/' % bands
                netpath = 'D:/ProjectData/BrandNewCTC/Records-Origin-Right/Bands-%d-Session-%d/' % (
                    bands, session)
                savepath = 'D:/ProjectData/BrandNewCTC/Server/Records-Result-Origin-Right/Bands-%d-Session-%d-%s/' % (
                    bands, session, gender)
                # if os.path.exists(savepath): continue
                # os.makedirs(savepath + 'Decode')
                # os.makedirs(savepath + 'Logits')
                # os.makedirs(savepath + 'SoftMax')

                # trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Loader(loadpath=loadpath, session=session)
                testData = numpy.load(loadpath + '%s-Session%d-Data.npy' % (gender, session))
                testLabel = numpy.load(loadpath + '%s-Session%d-Label.npy' % (gender, session))
                testSeq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (gender, session))

                for episode in range(100):
                    if os.path.exists(savepath + 'Decode/%04d.csv' % episode): continue
                    if not os.path.exists(netpath + '%04d-Network.index' % episode): continue
                    graph = tensorflow.Graph()
                    with graph.as_default():
                        classifier = CTC_BLSTM(trainData=None, trainLabel=None, trainSeqLength=None,
                                               featureShape=bands, numClass=5, graphRevealFlag=False, batchSize=32,
                                               startFlag=False)
                        classifier.Load(loadpath=netpath + '%04d-Network' % episode)
                        matrixDecode, matrixLogits, matrixSoftMax = classifier.Test_AllMethods(testData=testData,
                                                                                               testLabel=testLabel,
                                                                                               testSeq=testSeq)
                        # exit()
                        file = open(savepath + 'Decode/%04d.csv' % episode, 'w')
                        for indexX in range(4):
                            for indexY in range(4):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixDecode[indexX][indexY]))
                            file.write('\n')
                        file.close()
                        file = open(savepath + 'Logits/%04d.csv' % episode, 'w')
                        for indexX in range(4):
                            for indexY in range(4):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixLogits[indexX][indexY]))
                            file.write('\n')
                        file.close()
                        file = open(savepath + 'SoftMax/%04d.csv' % episode, 'w')
                        for indexX in range(4):
                            for indexY in range(4):
                                if indexY != 0: file.write(',')
                                file.write(str(matrixSoftMax[indexX][indexY]))
                            file.write('\n')
                        file.close()
