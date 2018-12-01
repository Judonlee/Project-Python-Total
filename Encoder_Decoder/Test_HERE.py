import tensorflow
from tensorflow.contrib import seq2seq
from __Base.BaseClass import NeuralNetwork_Base
import numpy


class EnDecoder(NeuralNetwork_Base):
    def __init__(self, encoderData, decoderData, batchSize=32, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.encoderData = encoderData
        self.decoderData = decoderData
        super(EnDecoder, self).__init__(trainData=None, trainLabel=None, batchSize=batchSize, learningRate=learningRate,
                                        startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath,
                                        occupyRate=occupyRate)
        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.EncoderDataInput = tensorflow.placeholder(dtype=tensorflow.float32,
                                                       shape=[None, None, numpy.shape(self.encoderData)[2]],
                                                       name='EncoderDataInput')
        self.EncoderLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[2], name='EncoderLenInput')
        self.DecoderDataInput = tensorflow.placeholder(dtype=tensorflow.float32,
                                                       shape=[None, None, numpy.shape(self.decoderData)[2]],
                                                       name='DecoderDataInput')
        self.DecoderLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[2], name='DecoderLenInput')

        self.parameters['RNN_Cell'] = tensorflow.nn.rnn_cell.BasicLSTMCell(num_units=128)
        self.parameters['RNN_Output'], self.parameters['RNN_FinalState'] = tensorflow.nn.dynamic_rnn(
            cell=self.parameters['RNN_Cell'], inputs=self.EncoderDataInput, sequence_length=self.EncoderLenInput,
            time_major=False, dtype=tensorflow.float32)

        self.parameters['Helper'] = seq2seq.TrainingHelper(inputs=self.DecoderDataInput,
                                                           sequence_length=self.DecoderLenInput, time_major=False)
        with tensorflow.variable_scope('Decoder'):
            self.parameters['Decoder_Cell'] = tensorflow.nn.rnn_cell.BasicLSTMCell(num_units=128)
            self.parameters['Decoder'] = seq2seq.BasicDecoder(cell=self.parameters['Decoder_Cell'],
                                                              helper=self.parameters['Helper'],
                                                              initial_state=self.parameters['RNN_FinalState'])
        self.parameters['Logits'], self.parameters['FinalState'], self.parameters[
            'FinalSequenceLen'] = seq2seq.dynamic_decode(decoder=self.parameters['Decoder'])

    def Valid(self):
        result = self.session.run(fetches=self.parameters['Logits'],
                                  feed_dict={self.EncoderDataInput: self.encoderData, self.EncoderLenInput: [3, 3],
                                             self.DecoderDataInput: self.decoderData, self.DecoderLenInput: [3, 2]})
        print(result)
        print(numpy.shape(result.rnn_output))


if __name__ == '__main__':
    # 早上好 [1,0,0,0,0],[0,1,0,0,0],[0,0,1,0,0]
    # 下午好 [0,0,0,1,0],[0,0,0,0,1],[0,0,1,0,0]
    encoderData = [[[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]],
                   [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]]]
    # Good Morning      [1,0,0],[0,1,0]
    # Good Afternoon    [1,0,0],[0,0,1]
    decoderData = [[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                   [[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]]

    # print(numpy.shape(encoderData),numpy.shape(decoderData))

    classifier = EnDecoder(encoderData=encoderData, decoderData=decoderData)
    print(classifier.information)
    classifier.Valid()
