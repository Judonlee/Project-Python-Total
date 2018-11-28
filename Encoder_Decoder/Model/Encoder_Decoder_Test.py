import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy
from tensorflow.contrib import seq2seq


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
        self.encoderDataInput = tensorflow.placeholder(dtype=tensorflow.float32,
                                                       shape=[None, None, numpy.shape(self.encoderData)[2]],
                                                       name='encoderDataInput')
        self.encoderSeqInput = tensorflow.placeholder(dtype=tensorflow.int64, name='encoderSeqInput')
        self.decoderDataInput = tensorflow.placeholder(dtype=tensorflow.float32,
                                                       shape=[None, None, numpy.shape(self.decoderData)[2]],
                                                       name='decoderDataInput')
        # self.decoderSeqInput = tensorflow.placeholder(dtype=tensorflow.int64, shape=[None], name='decoderSeqInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.encoderDataInput, name='BatchSize')[0]
        self.parameters['TimeStep'] = tensorflow.shape(input=self.encoderDataInput, name='TimeStep')[1]

        #####################################################################################################
        # Encoder Part Start
        #####################################################################################################

        self.parameters['RNN_Cell_Forward'] = []
        self.parameters['RNN_Cell_Backward'] = []

        for layers in range(self.rnnLayer):
            self.parameters['RNN_Cell_Forward'].append(
                tensorflow.nn.rnn_cell.LSTMCell(num_units=self.hiddenNodules, state_is_tuple=True,
                                                name='RNN_Cell_Forward_%d' % layers))
            self.parameters['RNN_Cell_Backward'].append(
                tensorflow.nn.rnn_cell.LSTMCell(num_units=self.hiddenNodules, state_is_tuple=True,
                                                name='RNN_Cell_Backward_%d' % layers))

        self.parameters['Layer_Forward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=self.parameters['RNN_Cell_Forward'], state_is_tuple=True)
        self.parameters['Layer_Backward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=self.parameters['RNN_Cell_Backward'], state_is_tuple=True)

        (self.parameters['RNN_Output_Forward'], self.parameters['RNN_Output_Backward']), (
            self.parameters['RNN_FinalState_Forward'], self.parameters['RNN_FinalState_Backward']) = \
            tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['Layer_Forward'],
                                                    cell_bw=self.parameters['Layer_Backward'],
                                                    inputs=self.encoderDataInput, sequence_length=self.encoderSeqInput,
                                                    dtype=tensorflow.float32)

        self.parameters['RNN_FinalState_C'] = tensorflow.concat(
            (self.parameters['RNN_FinalState_Forward'][0], self.parameters['RNN_FinalState_Backward'][0]), axis=1)
        self.parameters['RNN_FinalState_H'] = tensorflow.concat(
            (self.parameters['RNN_FinalState_Forward'][1], self.parameters['RNN_FinalState_Backward'][1]), axis=1)

    def Valid(self):
        result = self.session.run(fetches=self.parameters['RNN_FinalState_H'],
                                  feed_dict={self.encoderDataInput: self.encoderData,
                                             self.encoderSeqInput: self.encoderSeq})
        # print(result)
        print(numpy.shape(result))


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
    classifier = EncoderDecoder_Test(encoderData=InputEnglish, encoderSeq=[2, 2, 2], decoderData=InputChinese,
                                     decoderSeq=[3, 3, 3], rnnLayer=2)
    print(classifier.information)
    classifier.Valid()
