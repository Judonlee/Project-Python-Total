from __Base.BaseClass import NeuralNetwork_Base
import numpy
import tensorflow
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense

VOCABULAR = 41

NETWORK_LENGTH = {'SA': 53}


class AttentionTransform(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainDataSeq, trainLabelSeq, firstAttention=None, firstAttentionName=None,
                 firstAttentionScope=None, rnnLayer=2, featureShape=40, batchSize=32, hiddenNodules=128,
                 learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        #####################################################################
        if len(trainData) != len(trainLabel) or len(trainLabel) != len(trainDataSeq) or \
                len(trainDataSeq) != len(trainLabelSeq):
            raise RuntimeError("Train Data, Label, Seq don't have same Len!")
        #####################################################################

        self.dataSeq = trainDataSeq
        self.labelSeq = trainLabelSeq

        self.firstAttention = firstAttention
        self.firstAttentionName = firstAttentionName
        self.firstAttentionScope = firstAttentionScope

        self.rnnLayers = rnnLayer
        self.hiddenNodules = hiddenNodules
        self.featureShape = featureShape
        super(AttentionTransform, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        #############################################################################
        # Input Data
        #############################################################################

        self.dataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[None, None, self.featureShape], name='dataInput')
        self.labelInputSR = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None], name='labelInput')
        self.dataLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='dataLenInput')
        self.labelLenInputSR = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='labelLenInput')

        #############################################################################
        # Batch Parameters
        #############################################################################

        self.parameters['BatchSize'], self.parameters['TimeStep'], _ = tensorflow.unstack(
            tensorflow.shape(input=self.dataInput, name='DataShape'))
        self.parameters['LabelStep'] = tensorflow.shape(input=self.labelInputSR, name='LabelShape')[1]

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
                    inputs=self.dataInput, sequence_length=self.dataLenInput, dtype=tensorflow.float32)

        if self.firstAttention is None:
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
            self.attentionList = self.firstAttention(dataInput=self.parameters['Encoder_Output'],
                                                     scopeName=self.firstAttentionName,
                                                     hiddenNoduleNumber=2 * self.hiddenNodules,
                                                     attentionScope=self.firstAttentionScope, blstmFlag=True)
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

        self.parameters['DecoderEmbedding'] = tensorflow.Variable(
            initial_value=tensorflow.truncated_normal(shape=[VOCABULAR, self.hiddenNodules * 2], stddev=0.1,
                                                      name='DecoderEmbedding'))

        self.parameters['DecoderEmbeddingResult'] = tensorflow.nn.embedding_lookup(
            params=self.parameters['DecoderEmbedding'], ids=self.labelInputSR, name='DecoderEmbeddingResult')

        #############################################################################
        # Decoder
        #############################################################################

        self.parameters['Decoder_Helper'] = seq2seq.TrainingHelper(
            inputs=self.parameters['DecoderEmbeddingResult'], sequence_length=self.labelLenInputSR,
            name='Decoder_Helper')
        with tensorflow.variable_scope('Decoder'):
            self.parameters['Decoder_FC'] = Dense(VOCABULAR)

            self.parameters['Decoder_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules * 2) for _ in range(self.rnnLayers)],
                state_is_tuple=True)

            self.parameters['Decoder'] = seq2seq.BasicDecoder(cell=self.parameters['Decoder_Cell'],
                                                              helper=self.parameters['Decoder_Helper'],
                                                              initial_state=self.parameters['Decoder_InitalState'],
                                                              output_layer=self.parameters['Decoder_FC'])

            self.parameters['Decoder_Logits'], self.parameters['Decoder_FinalState'], self.parameters[
                'Decoder_FinalSeq'] = seq2seq.dynamic_decode(decoder=self.parameters['Decoder'])

        with tensorflow.name_scope('Loss'):
            self.parameters['TargetsReshape'] = tensorflow.reshape(tensor=self.labelInputSR, shape=[-1],
                                                                   name='TargetsReshape')
            self.parameters['Decoder_Reshape'] = tensorflow.reshape(self.parameters['Decoder_Logits'].rnn_output,
                                                                    [-1, VOCABULAR], name='Decoder_Reshape')
            self.parameters['Cost'] = tensorflow.losses.sparse_softmax_cross_entropy(
                labels=self.parameters['TargetsReshape'], logits=self.parameters['Decoder_Reshape'])

            self.trainEncoderDecoder = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
                self.parameters['Cost'])

    def Train(self):
        trainData, trainlabel, dataSeq, labelSeq = self.data, self.label, self.dataSeq, self.labelSeq

        startPosition = 0
        totalLoss = 0.0
        while startPosition < len(trainData):
            batchData = []
            batchLabel = []
            batchDataSeq = dataSeq[startPosition:startPosition + self.batchSize]
            batchLabelSeq = labelSeq[startPosition:startPosition + self.batchSize]

            for index in range(min(self.batchSize, len(trainData) - startPosition)):
                currentData = numpy.concatenate([trainData[startPosition + index], numpy.zeros(
                    shape=[max(batchDataSeq) - numpy.shape(trainData[index + startPosition])[0],
                           numpy.shape(trainData[index])[1]])], axis=0)
                batchData.append(currentData)

            for index in range(min(self.batchSize, len(trainData) - startPosition)):
                currentLabel = numpy.concatenate([trainlabel[startPosition + index], numpy.zeros(
                    shape=[max(batchLabelSeq) - numpy.shape(trainlabel[index + startPosition])[0]])], axis=0)
                # print(len(trainlabel[startPosition + index]), len(currentLabel))
                batchLabel.append(currentLabel)

            # print(numpy.shape(batchData), numpy.shape(batchLabel))
            loss = self.session.run(fetches=self.parameters['Cost'],
                                    feed_dict={self.dataInput: batchData, self.labelInputSR: batchLabel,
                                               self.dataLenInput: batchDataSeq, self.labelLenInputSR: batchLabelSeq})

            totalLoss += loss
            print('\rTraining : %d/%d\tLoss = %f' % (startPosition, len(trainData), loss), end='')
            startPosition += self.batchSize

        return totalLoss

    def LoadPart(self, loadpath):
        pass
