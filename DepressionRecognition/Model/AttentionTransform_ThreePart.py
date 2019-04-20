from __Base.BaseClass import NeuralNetwork_Base
import numpy
import tensorflow
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense
from __Base.Shuffle import Shuffle_Five

VOCABULAR = 41
NETWORK_LENGTH = {'SA': 53, 'LA': 53, 'MA': 59, 'MCA': 53}
TRANSFORM_NAME = {'SA': 'AttentionFinal', 'LA': 'AttentionFinal', 'MA': 'Probability_Result',
                  'MCA': 'AttentionProbability'}


class AttentionTransform_ThreePart(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainDataSeq, trainLabelSeq, sentenceLevel, speechLevel, firstAttention,
                 firstAttentionName, firstAttentionScope, secondAttention=None, secondAttentionName=None,
                 secondAttentionScope=None, rnnLayer=2, featureShape=40, batchSize=32, hiddenNodules=128,
                 learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1,
                 attentionTransformLoss='L1', attentionTransformWeight=1, lossType='RMSE'):
        #####################################################################
        # if len(trainData) != len(trainLabel) or len(trainLabel) != len(trainDataSeq):
        #     raise RuntimeError("Train Data, Label, Seq don't have same Len!")
        #####################################################################

        self.dataSeq = trainDataSeq
        self.labelSeq = trainLabelSeq

        self.sentenceData = sentenceLevel
        self.speechData = speechLevel

        self.firstAttention = firstAttention
        self.firstAttentionName = firstAttentionName
        self.firstAttentionScope = firstAttentionScope
        self.secondAttention = secondAttention
        self.secondAttentionName = secondAttentionName
        self.secondAttentionScope = secondAttentionScope

        self.rnnLayers = rnnLayer
        self.hiddenNodules = hiddenNodules
        self.featureShape = featureShape

        self.attentionTransformLoss = attentionTransformLoss
        self.attentionTransformWeight = attentionTransformWeight
        self.lossType = lossType

        super(AttentionTransform_ThreePart, self).__init__(
            trainData=trainData, trainLabel=trainLabel, batchSize=batchSize, learningRate=learningRate,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        #############################################################################
        # Input Data
        #############################################################################

        self.dataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[None, None, self.featureShape], name='dataInput')
        self.dataLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='dataLenInput')

        self.labelInputSR = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None], name='labelInput')
        self.labelLenInputSR = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='labelLenInput')

        self.labelInputDR = tensorflow.placeholder(dtype=tensorflow.float32, shape=None, name='labelInputDR')

        self.sentenceDataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[None, 2 * self.hiddenNodules], name='sentenceDataInput')
        self.speechDataInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[1, 2 * self.hiddenNodules], name='speechDataInput')

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

        #############################################################################
        self.DBLSTM_Structure(learningRate=learningRate)

    def DBLSTM_Structure(self, learningRate):
        with tensorflow.variable_scope('First_BLSTM_DR'):
            self.parameters['First_FW_Cell_DR'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
            self.parameters['First_BW_Cell_DR'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['First_Output_DR'], self.parameters['First_FinalState_DR'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['First_FW_Cell_DR'],
                                                        cell_bw=self.parameters['First_BW_Cell_DR'],
                                                        inputs=self.dataInput, sequence_length=self.dataLenInput,
                                                        dtype=tensorflow.float32)

        self.firstAttentionList_DR = self.firstAttention(dataInput=self.parameters['First_Output_DR'],
                                                         scopeName=self.firstAttentionName + '_SR',
                                                         hiddenNoduleNumber=2 * self.hiddenNodules,
                                                         attentionScope=self.firstAttentionScope, blstmFlag=True)

        self.parameters['First_FinalOutput_DR'] = tensorflow.concat(
            [self.firstAttentionList_DR['FinalResult'], self.sentenceDataInput], axis=1, name='First_FinalOutput_DR')

        with tensorflow.variable_scope('TransformLoss'):
            if self.attentionTransformLoss == 'L1':
                self.parameters['AttentionLoss'] = self.attentionTransformWeight * tensorflow.reduce_mean(
                    input_tensor=tensorflow.abs(self.attentionList[TRANSFORM_NAME[self.firstAttentionName]] -
                                                self.firstAttentionList_DR[TRANSFORM_NAME[self.firstAttentionName]]),
                    name='AttentionLoss')
            if self.attentionTransformLoss == 'L2':
                self.parameters['AttentionLoss'] = self.attentionTransformWeight * tensorflow.nn.l2_loss(
                    t=tensorflow.abs(self.attentionList[TRANSFORM_NAME[self.firstAttentionName]] -
                                     self.firstAttentionList_DR[TRANSFORM_NAME[self.firstAttentionName]]),
                    name='AttentionLoss')

        with tensorflow.variable_scope('Second_BLSTM_DR'):
            self.parameters['Second_FW_Cell_DR'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)],
                state_is_tuple=True)
            self.parameters['Second_BW_Cell_DR'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)],
                state_is_tuple=True)

            self.parameters['Second_Output_DR'], self.parameters['Second_FinalState_DR'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.parameters['Second_FW_Cell_DR'], cell_bw=self.parameters['Second_BW_Cell_DR'],
                    inputs=self.parameters['First_FinalOutput_DR'][tensorflow.newaxis, :, :],
                    dtype=tensorflow.float32)

            ##########################################################################

        if self.secondAttention is None:
            self.parameters['Second_FinalOutput_DR'] = tensorflow.concat(
                [self.parameters['Second_FinalState_DR'][self.rnnLayers - 1][0].h,
                 self.parameters['Second_FinalState_DR'][self.rnnLayers - 1][1].h], axis=1)
        else:
            self.secondAttentionList_DR = self.secondAttention(dataInput=self.parameters['Second_Output_DR'],
                                                               scopeName=self.secondAttentionName,
                                                               hiddenNoduleNumber=2 * self.hiddenNodules,
                                                               attentionScope=self.secondAttentionScope,
                                                               blstmFlag=True)
            self.parameters['Second_FinalOutput_DR'] = self.secondAttentionList_DR['FinalResult']

        self.parameters['Second_FinalOutput_DR'] = tensorflow.concat(
            [self.parameters['Second_FinalOutput_DR'], self.speechDataInput], axis=1, name='Second_FinalOutput_DR')

        self.parameters['FinalPredict_DR'] = tensorflow.reshape(
            tensor=tensorflow.layers.dense(inputs=self.parameters['Second_FinalOutput_DR'], units=1,
                                           activation=None, name='FinalPredict_DR'), shape=[1])

        if self.lossType == 'MSE':
            self.parameters['Loss_DR'] = tensorflow.losses.mean_squared_error(
                labels=self.labelInputDR, predictions=self.parameters['FinalPredict_DR'])
        if self.lossType == 'RMSE':
            self.parameters['Loss_DR'] = tensorflow.sqrt(
                tensorflow.losses.mean_squared_error(labels=self.labelInputDR,
                                                     predictions=self.parameters['FinalPredict_DR']))
        if self.lossType == 'MAE':
            self.parameters['Loss_DR'] = tensorflow.losses.absolute_difference(
                labels=self.labelInputDR, predictions=self.parameters['FinalPredict_DR'])
        self.train_DR = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            loss=self.parameters['Loss_DR'] + self.parameters['AttentionLoss'],
            var_list=tensorflow.global_variables()[NETWORK_LENGTH[self.firstAttentionName]:])

    def Load_AttentionTransform(self, loadpath):
        saver = tensorflow.train.Saver(tensorflow.global_variables()[0:NETWORK_LENGTH[self.firstAttentionName]])
        saver.restore(self.session, loadpath)

    def Train(self, logName):
        trainData, trainLabel, trainSeq, sentenceData, speechData = \
            Shuffle_Five(self.data, self.label, self.dataSeq, self.sentenceData, self.speechData)
        # print(numpy.shape(sentenceData[0]), numpy.shape(speechData[0]))
        totalLoss = 0
        with open(logName, 'w') as file:
            for index in range(len(trainData)):
                distanceLoss, attentionLoss, _ = self.session.run(
                    fetches=[self.parameters['Loss_DR'], self.parameters['AttentionLoss'], self.train_DR],
                    feed_dict={self.dataInput: trainData[index], self.labelInputDR: trainLabel[index],
                               self.dataLenInput: trainSeq[index], self.sentenceDataInput: sentenceData[index],
                               self.speechDataInput: speechData[index]})
                totalLoss += distanceLoss + attentionLoss
                file.write(str(distanceLoss) + ',' + str(attentionLoss) + '\n')
                print('\rTraining %d/%d DistanceLoss = %f\tAttentionLoss = %f' % (
                    index, len(trainData), distanceLoss, attentionLoss), end='')
        return totalLoss

    def Test(self, logName, testData, testLabel, testSeq, testSentence, testSpeech):
        with open(logName, 'w') as file:
            for index in range(len(testData)):
                predict = self.session.run(
                    fetches=self.parameters['FinalPredict_DR'],
                    feed_dict={self.dataInput: testData[index], self.dataLenInput: testSeq[index],
                               self.sentenceDataInput: testSentence[index], self.speechDataInput: testSpeech[index]})
                file.write(str(testLabel[index]) + ',' + str(predict) + '\n')
                print('\rTraining %d/%d' % (index, len(testData)), end='')
