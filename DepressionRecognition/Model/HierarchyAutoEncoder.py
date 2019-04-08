import tensorflow
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import seq2seq
import numpy
from __Base.BaseClass import NeuralNetwork_Base
from __Base.Shuffle import Shuffle


class HierarchyAutoEncoder(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeq, firstAttention=None, firstAttentionName='',
                 firstAttentionScope=0, secondAttention=None, secondAttentionName='', secondAttentionScope=0,
                 rnnLayers=2, hiddenNoduleNumber=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.seq = trainSeq

        self.firstAttention = firstAttention
        self.firstAttentionName = firstAttentionName
        self.firstAttentionScope = firstAttentionScope
        self.secondAttention = secondAttention
        self.secondAttentionName = secondAttentionName
        self.secondAttentionScope = secondAttentionScope

        self.rnnLayers = rnnLayers
        self.hiddenNodules = hiddenNoduleNumber

        super(HierarchyAutoEncoder, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=-1, learningRate=learningRate, startFlag=startFlag,
            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, 40],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=None, name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqInput')

        ##########################################################################

        self.parameters['BatchSize'], self.parameters['TimeStep'], _ = tensorflow.unstack(
            tensorflow.shape(input=self.dataInput, name='Parameter'))

        ##########################################################################

        with tensorflow.variable_scope('First_BLSTM'):
            self.parameters['First_FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
            self.parameters['First_BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['First_Output'], self.parameters['First_FinalState'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['First_FW_Cell'],
                                                        cell_bw=self.parameters['First_BW_Cell'], inputs=self.dataInput,
                                                        sequence_length=self.seqInput, dtype=tensorflow.float32)

        if self.firstAttention is None:
            self.parameters['First_FinalOutput'] = tensorflow.concat(
                [self.parameters['First_FinalState'][self.rnnLayers - 1][0].h,
                 self.parameters['First_FinalState'][self.rnnLayers - 1][1].h], axis=1)
        else:
            self.firstAttentionList = self.firstAttention(dataInput=self.parameters['First_Output'],
                                                          scopeName=self.firstAttentionName,
                                                          hiddenNoduleNumber=2 * self.hiddenNodules,
                                                          attentionScope=self.firstAttentionScope, blstmFlag=True)
            self.parameters['First_FinalOutput'] = self.firstAttentionList['FinalResult']

        ##########################################################################

        with tensorflow.variable_scope('Second_BLSTM'):
            self.parameters['Second_FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)],
                state_is_tuple=True)
            self.parameters['Second_BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)],
                state_is_tuple=True)

            self.parameters['Second_Output'], self.parameters['Second_FinalState'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.parameters['Second_FW_Cell'], cell_bw=self.parameters['Second_BW_Cell'],
                    inputs=self.parameters['First_FinalOutput'][tensorflow.newaxis, :, :],
                    dtype=tensorflow.float32)

        ##########################################################################

        if self.secondAttention is None:
            self.parameters['Decoder_InitalState'] = []
            for index in range(self.rnnLayers):
                self.parameters['Encoder_Cell_Layer%d' % index] = rnn.LSTMStateTuple(
                    c=tensorflow.concat([self.parameters['Second_FinalState'][index][0].c,
                                         self.parameters['Second_FinalState'][index][1].c], axis=1),
                    h=tensorflow.concat([self.parameters['Second_FinalState'][index][0].h,
                                         self.parameters['Second_FinalState'][index][1].h], axis=1))
                self.parameters['Decoder_InitalState'].append(self.parameters['Encoder_Cell_Layer%d' % index])
            self.parameters['Decoder_InitalState_First'] = tuple(self.parameters['Decoder_InitalState'])

        #########################################################################

        with tensorflow.variable_scope('Decoder_First'):
            self.parameters['Decoder_Helper_First'] = seq2seq.TrainingHelper(
                inputs=self.parameters['First_FinalOutput'][tensorflow.newaxis, :, :],
                sequence_length=[self.parameters['BatchSize']],
                name='Decoder_Helper_First')
            self.parameters['Decoder_FC_First'] = Dense(self.hiddenNodules * 2)
            self.parameters['Decoder_Cell_First'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules * 2) for _ in range(self.rnnLayers)],
                state_is_tuple=True)

            self.parameters['Decoder_First'] = seq2seq.BasicDecoder(
                cell=self.parameters['Decoder_Cell_First'], helper=self.parameters['Decoder_Helper_First'],
                initial_state=self.parameters['Decoder_InitalState_First'],
                output_layer=self.parameters['Decoder_FC_First'])

            self.parameters['Decoder_Logits_First'], self.parameters['Decoder_FinalState_First'], self.parameters[
                'Decoder_FinalSeq_First'] = seq2seq.dynamic_decode(decoder=self.parameters['Decoder_First'])

        self.parameters['Decoder_First_Result'] = self.parameters['Decoder_Logits_First'][0]

        self.parameters['Decoder_InitialState_Second_Media'] = []
        for index in range(self.rnnLayers):
            self.parameters['Decoder_Cell_Second_Layer%d' % index] = rnn.LSTMStateTuple(
                c=self.parameters['Decoder_First_Result'][0], h=self.parameters['Decoder_First_Result'][0])
            self.parameters['Decoder_InitialState_Second_Media'].append(
                self.parameters['Decoder_Cell_Second_Layer%d' % index])
        self.parameters['Decoder_InitialState_Second'] = tuple(self.parameters['Decoder_InitialState_Second_Media'])

        with tensorflow.variable_scope('Decoder_Second'):
            self.parameters['Decoder_Helper_Second'] = seq2seq.TrainingHelper(
                inputs=self.dataInput, sequence_length=self.seqInput, name='Decoder_Helper_Second')
            self.parameters['Decoder_FC_Second'] = Dense(40)
            self.parameters['Decoder_Cell_Second'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules * 2) for _ in range(self.rnnLayers)],
                state_is_tuple=True)

            self.parameters['Decoder_Second'] = seq2seq.BasicDecoder(
                cell=self.parameters['Decoder_Cell_Second'], helper=self.parameters['Decoder_Helper_Second'],
                initial_state=self.parameters['Decoder_InitialState_Second'],
                output_layer=self.parameters['Decoder_FC_Second'])

            self.parameters['Decoder_Logits_Second'], self.parameters['Decoder_FinalState_Second'], self.parameters[
                'Decoder_FinalSeq_Second'] = seq2seq.dynamic_decode(decoder=self.parameters['Decoder_Second'])

    def Train(self):
        # trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seq)
        trainData, trainLabel, trainSeq = self.data, self.label, self.seq
        for index in range(len(trainData)):
            loss = self.session.run(fetches=self.parameters['Decoder_Logits_Second'][0],
                                    feed_dict={self.dataInput: trainData[index], self.labelInput: trainLabel[index],
                                               self.seqInput: trainSeq[index]})
            print(loss)
            print(numpy.shape(loss))
            exit()
