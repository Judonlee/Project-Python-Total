from __Base.BaseClass import NeuralNetwork_Base
from DepressionRecognition.Tools import Shuffle_Part4
import numpy
import tensorflow
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
import os

VOCABULAR = 41


class EncoderDecoder_Base(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainDataSeq, trainLabelSeq, attention=None, attentionName=None,
                 attentionScope=None, rnnLayer=2, featureShape=40, batchSize=32, hiddenNodules=128, learningRate=1E-3,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        #####################################################################
        if len(trainData) != len(trainLabel) or len(trainLabel) != len(trainDataSeq) or \
                len(trainDataSeq) != len(trainLabelSeq):
            raise RuntimeError("Train Data, Label, Seq don't have same Len!")
        #####################################################################

        self.dataSeq = trainDataSeq
        self.labelSeq = trainLabelSeq

        self.attention = attention
        self.attentionName = attentionName
        self.attentionScope = attentionScope

        self.rnnLayers = rnnLayer
        self.hiddenNodules = hiddenNodules
        self.featureShape = featureShape
        super(EncoderDecoder_Base, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        #############################################################################
        # Input Data
        #############################################################################

        self.dataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[None, None, self.featureShape], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None], name='labelInput')
        self.dataLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='dataLenInput')
        self.labelLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='labelLenInput')

        #############################################################################
        # Batch Parameters
        #############################################################################

        self.parameters['BatchSize'], self.parameters['TimeStep'], _ = tensorflow.unstack(
            tensorflow.shape(input=self.dataInput, name='DataShape'))
        self.parameters['LabelStep'] = tensorflow.shape(input=self.labelInput, name='LabelShape')[1]

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

        if self.attention is None:
            self.parameters['Decoder_InitalState'] = self.parameters['Encoder_FinalState']
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

        self.parameters['DecoderEmbedding'] = tensorflow.Variable(
            initial_value=tensorflow.truncated_normal(shape=[VOCABULAR, self.hiddenNodules * 2], stddev=0.1,
                                                      name='DecoderEmbedding'))

        self.parameters['DecoderEmbeddingResult'] = tensorflow.nn.embedding_lookup(
            params=self.parameters['DecoderEmbedding'], ids=self.labelInput, name='DecoderEmbeddingResult')

        #############################################################################
        # Decoder
        #############################################################################

        self.parameters['Decoder_Helper'] = seq2seq.TrainingHelper(
            inputs=self.parameters['DecoderEmbeddingResult'], sequence_length=self.labelLenInput, name='Decoder_Helper')
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
            self.parameters['TargetsReshape'] = tensorflow.reshape(tensor=self.labelInput, shape=[-1],
                                                                   name='TargetsReshape')
            self.parameters['Decoder_Reshape'] = tensorflow.reshape(self.parameters['Decoder_Logits'].rnn_output,
                                                                    [-1, VOCABULAR], name='Decoder_Reshape')
            self.parameters['Cost'] = tensorflow.losses.sparse_softmax_cross_entropy(
                labels=self.parameters['TargetsReshape'], logits=self.parameters['Decoder_Reshape'])

            self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Cost'])

    def ValidTest(self):
        # trainData, trainlabel, dataSeq, labelSeq = Shuffle_Part4(
        #     data=self.data, label=self.label, dataLen=self.dataSeq, labelLen=self.labelSeq)
        trainData, trainlabel, dataSeq, labelSeq = self.data, self.label, self.dataSeq, self.labelSeq

        startPosition = 0
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
                batchLabel.append(currentLabel)

            loss = self.session.run(fetches=self.parameters['Cost'],
                                    feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                               self.dataLenInput: batchDataSeq, self.labelLenInput: batchLabelSeq})
            print(loss)
            # print(numpy.shape(loss))
            exit()

    def Train(self, logName):
        with open(logName, 'w') as file:
            trainData, trainlabel, dataSeq, labelSeq = Shuffle_Part4(
                data=self.data, label=self.label, dataLen=self.dataSeq, labelLen=self.labelSeq)

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
                loss, _ = self.session.run(fetches=[self.parameters['Cost'], self.train],
                                           feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                      self.dataLenInput: batchDataSeq,
                                                      self.labelLenInput: batchLabelSeq})

                file.write(str(loss) + '\n')
                totalLoss += loss
                print('\rTraining : %d/%d\tLoss = %f' % (startPosition, len(trainData), loss), end='')
                startPosition += self.batchSize
        return totalLoss
