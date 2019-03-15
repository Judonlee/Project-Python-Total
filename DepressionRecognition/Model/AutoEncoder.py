import tensorflow
import numpy
from __Base.BaseClass import NeuralNetwork_Base
from __Base.Shuffle import Shuffle_Train
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense


class AutoEncoder(NeuralNetwork_Base):
    def __init__(self, data, seq, weight=10, attention=None, attentionName=None, attentionScope=None, batchSize=32,
                 maxLen=1000, rnnLayers=2, hiddenNodules=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.treatData = data
        self.treatSeq = seq
        self.weight = weight

        self.attention = attention
        self.attentionName = attentionName
        self.attentionScope = attentionScope

        self.maxLen = maxLen
        self.rnnLayers = rnnLayers
        self.hiddenNodules = hiddenNodules
        self.featureShape = numpy.shape(data[0])[1]
        # print(self.featureShape)

        super(AutoEncoder, self).__init__(trainData=None, trainLabel=None, batchSize=batchSize,
                                          learningRate=learningRate, startFlag=startFlag,
                                          graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='DataInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='SeqInput')

        #############################################################################
        # Batch Parameters
        #############################################################################

        self.parameters['BatchSize'], self.parameters['TimeStep'], _ = tensorflow.unstack(
            tensorflow.shape(input=self.dataInput, name='DataShape'))

        ###################################################################################################
        # Encoder
        ###################################################################################################

        with tensorflow.variable_scope('Encoder'):
            self.parameters['Encoder_Cell_Forward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
            self.parameters['Encoder_Cell_Backward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['Encoder_Output'], self.parameters['Encoder_FinalState'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.parameters['Encoder_Cell_Forward'], cell_bw=self.parameters['Encoder_Cell_Backward'],
                    inputs=self.dataInput, sequence_length=self.seqInput, dtype=tensorflow.float32)

        if self.attention is None:
            self.parameters['Decoder_InitalState'] = []
            for index in range(self.rnnLayers):
                self.parameters['Encoder_Cell_Layer%d' % index] = rnn.LSTMStateTuple(
                    c=tensorflow.concat([self.parameters['Encoder_FinalState'][index][0].c,
                                         self.parameters['Encoder_FinalState'][index][1].c], axis=1),
                    h=tensorflow.concat([self.parameters['Encoder_FinalState'][index][0].h,
                                         self.parameters['Encoder_FinalState'][index][1].h], axis=1))
                self.parameters['Decoder_InitalState'].append(self.parameters['Encoder_Cell_Layer%d' % index])
            self.parameters['Decoder_InitalState'] = tuple(self.parameters['Decoder_InitalState'])
        else:
            self.attentionList = self.attention(dataInput=self.parameters['Encoder_Output'],
                                                scopeName=self.attentionName, hiddenNoduleNumber=2 * self.hiddenNodules,
                                                attentionScope=self.attentionScope, blstmFlag=True)
            self.parameters['Decoder_InitalState'] = []
            for index in range(self.rnnLayers):
                self.parameters['Encoder_Cell_Layer%d' % index] = rnn.LSTMStateTuple(
                    c=self.attentionList['FinalResult'],
                    h=tensorflow.concat(
                        [self.parameters['Encoder_FinalState'][index][0].h,
                         self.parameters['Encoder_FinalState'][index][1].h],
                        axis=1))
                self.parameters['Decoder_InitalState'].append(self.parameters['Encoder_Cell_Layer%d' % index])
            self.parameters['Decoder_InitalState'] = tuple(self.parameters['Decoder_InitalState'])

        #############################################################################
        # Decoder Label Pretreatment
        #############################################################################

        self.parameters['Decoder_Helper'] = seq2seq.TrainingHelper(
            inputs=self.dataInput, sequence_length=self.seqInput, name='Decoder_Helper')
        with tensorflow.variable_scope('Decoder'):
            self.parameters['Decoder_FC'] = Dense(self.featureShape)
            self.parameters['Decoder_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules * 2) for _ in range(self.rnnLayers)],
                state_is_tuple=True)

            self.parameters['Decoder'] = seq2seq.BasicDecoder(cell=self.parameters['Decoder_Cell'],
                                                              helper=self.parameters['Decoder_Helper'],
                                                              initial_state=self.parameters['Decoder_InitalState'],
                                                              output_layer=self.parameters['Decoder_FC'])

            self.parameters['Decoder_Logits'], self.parameters['Decoder_FinalState'], self.parameters[
                'Decoder_FinalSeq'] = seq2seq.dynamic_decode(decoder=self.parameters['Decoder'])

        #############################################################################
        # Losses
        #############################################################################

        self.parameters['Loss'] = tensorflow.losses.absolute_difference(
            labels=self.dataInput, predictions=self.parameters['Decoder_Logits'][0], weights=self.weight)
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        data, seq = Shuffle_Train(data=self.treatData, label=self.treatSeq)
        # data, seq = self.treatData, self.treatSeq

        # for index in range(len(data)):
        #     print(numpy.shape(data[index]), seq[index])

        startPosition = 0
        totalLoss = 0.0
        while startPosition < numpy.shape(data)[0]:
            batchData, batchSeq = [], []

            maxSeq = min(numpy.max(seq[startPosition:startPosition + self.batchSize]), self.maxLen)
            for index in range(startPosition, min(startPosition + self.batchSize, numpy.shape(data)[0])):
                batchSeq.append(min(seq[index], self.maxLen))

            for index in range(startPosition, min(startPosition + self.batchSize, numpy.shape(data)[0])):
                if numpy.shape(data[index])[0] >= self.maxLen:
                    currentData = data[index][0:self.maxLen]
                else:
                    currentData = numpy.concatenate(
                        [data[index], numpy.zeros([maxSeq - numpy.shape(data[index])[0], numpy.shape(data[index])[1]])],
                        axis=0)
                # print(numpy.shape(currentData))
                batchData.append(currentData)

            # print(startPosition, numpy.shape(batchData), numpy.shape(batchSeq), maxSeq)
            # print(batchSeq, '\n')

            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                       feed_dict={self.dataInput: batchData, self.seqInput: batchSeq})
            totalLoss += loss
            print('\rBatch %d/%d Loss = %f' % (startPosition, len(data), loss), end='')

            # exit()

            startPosition += self.batchSize
        return totalLoss
