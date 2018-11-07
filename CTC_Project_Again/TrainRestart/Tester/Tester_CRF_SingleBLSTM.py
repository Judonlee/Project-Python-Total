import tensorflow
import numpy
from CTC_Project_Again.ModelNew.Single_BLSTM_CTC_CRF import BLSTM_CTC_CRF
import os

if __name__ == '__main__':
    part = 'UA'
    for bands in [30]:
        for session in range(1, 6):
            for gender in ['Female', 'Male']:
                loadpath = 'D:/ProjectData/IEMOCAP-New/Bands%d/' % bands
                netpath = 'D:/ProjectData/BrandNewCTC/Single_BLSTM_CTC_CRF_%s/Bands-%d-Session-%d-%s/' % (
                    part, bands, session, gender)

                for testGender in ['Female', 'Male']:
                    if testGender == gender: continue
                    savepath = 'Result_Single_BLSTM_CTC_CRF_%s_Again/Bands-%d-Session-%d-%s-%s/' % (
                        part, bands, session, gender, testGender)

                    if os.path.exists(savepath): continue
                    os.makedirs(savepath)

                    testData = numpy.load(loadpath + '%s-Session%d-Data.npy' % (testGender, session))
                    testLabel = numpy.load(loadpath + '%s-Session%d-Label.npy' % (testGender, session))
                    testSeq = numpy.load(loadpath + '%s-Session%d-Seq.npy' % (testGender, session))

                    for episode in range(100):
                        if not os.path.exists(netpath + '%04d-Network.meta' % episode): continue
                        graph = tensorflow.Graph()
                        with graph.as_default():
                            classifier = BLSTM_CTC_CRF(trainData=None, trainLabel=None, trainSeqLength=None,
                                                       featureShape=bands, numClass=5, graphRevealFlag=False,
                                                       batchSize=32, startFlag=False)
                            classifier.Load(loadpath=netpath + '%04d-Network' % episode)
                            matrix = classifier.Test_CRF(testData=testData, testLabel=testLabel, testSeq=testSeq)

                            file = open(savepath + '%04d.csv' % episode, 'w')
                            for indexX in range(4):
                                for indexY in range(4):
                                    if indexY != 0: file.write(',')
                                    file.write(str(matrix[indexX][indexY]))
                                file.write('\n')
                            file.close()
                            # exit()
