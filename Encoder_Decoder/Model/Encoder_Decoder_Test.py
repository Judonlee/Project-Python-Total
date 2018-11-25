import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy


class EncoderDecoder_Test(NeuralNetwork_Base):
    def __init__(self, encoderData, encoderSeq, decoderData, decoderSeq, rnnLayer, hiddenNodules=128,
                 batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        self.encoderData = encoderData
        self.encoderSeq = encoderSeq
        self.decoderData = decoderData
        self.decoderSeq = decoderSeq
        self.rnnLayer = rnnLayer
        self.hiddenNodules = hiddenNodules

        super(EncoderDecoder_Test, self).__init__(trainData=None, trainLabel=None, batchSize=batchSize,
                                                  learningRate=learningRate, startFlag=startFlag,
                                                  graphRevealFlag=graphRevealFlag, graphPath=graphPath,
                                                  occupyRate=occupyRate)
        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + '\t' + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.encoderDataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, None],
                                                       name='encoderDataInput')
        self.encoderSeqInput = tensorflow.placeholder(dtype=tensorflow.int64, shape=[None], name='encoderSeqInput')
        self.decoderDataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, None],
                                                       name='decoderDataInput')
        self.decoderSeqInput = tensorflow.placeholder(dtype=tensorflow.int64, shape=[None], name='decoderSeqInput')



        # self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]
        # self.parameters['TimeStep'] = tensorflow.shape(input=self.dataInput, name='TimeStep')[1]


if __name__ == '__main__':
    # Good Morning [1,0,0,0] [0,1,0,0]
    # Good Afternoon [1,0,0,0] [0,0,1,0]
    # Good Evening [1,0,0,0] [0,0,0,1]
    InputEnglish = [[[1, 0, 0, 0], [0, 1, 0, 0]], [[1, 0, 0, 0], [0, 0, 1, 0]], [[1, 0, 0, 0], [0, 0, 0, 1]]]

    # 早上好[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,1,0,0,0]
    # 下午好[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,1,0,0,0]
    # 晚上好[0,0,0,0,0,1],[0,1,0,0,0,0],[0,0,1,0,0,0]
    InputChinese = [[[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]],
                    [[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 1, 0, 0, 0]],
                    [[0, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0]]]
    print(numpy.shape(InputEnglish), numpy.shape(InputChinese))
