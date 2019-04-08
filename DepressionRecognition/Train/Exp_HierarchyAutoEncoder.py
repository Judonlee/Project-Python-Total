from DepressionRecognition.Loader import Load_DBLSTM
from DepressionRecognition.Model.HierarchyAutoEncoder import HierarchyAutoEncoder
import numpy

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()

    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq), numpy.shape(testData),
          numpy.shape(testLabel), numpy.shape(testSeq))

    classifier = HierarchyAutoEncoder(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq)
    classifier.Train()
