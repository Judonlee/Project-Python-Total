import tensorflow
import numpy
from DepressionRecognition.Model.AutoEncoder import AutoEncoder
from __Base.Shuffle import Shuffle_Train
from DepressionRecognition.Tools import Shuffle_Part3
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense

NETWORK_LENGTH = {'None': 44, 'SA': 50, 'LA': 50, 'MA': 56, 'MCA': 50}


class DBLSTM_AutoEncoder(AutoEncoder):
    def __init__(self, data, label, seq, concatType='Concat', lossType='MAE', autoEncoderWeight=10, regressionWeight=24,
                 attention=None, attentionName=None, attentionScope=None, secondAttention=None,
                 secondAttentionName=None, secondAttentionScope=None, batchSize=32, maxLen=1000, rnnLayers=2,
                 hiddenNodules=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        self.concatType = concatType
        self.treatLabel = label
        self.lossType = lossType
        self.regressionWeight = regressionWeight

        self.secondAttention = secondAttention
        self.secondAttentionName = secondAttentionName
        self.secondAttentionScope = secondAttentionScope

        super(DBLSTM_AutoEncoder, self).__init__(
            data=data, seq=seq, weight=autoEncoderWeight, attention=attention, attentionName=attentionName,
            attentionScope=attentionScope, batchSize=batchSize, maxLen=maxLen, rnnLayers=rnnLayers,
            hiddenNodules=hiddenNodules, learningRate=learningRate, startFlag=startFlag,
            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, 40],
                                                name='DataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, name='LabelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='SeqInput')

        #############################################################################
        # Batch Parameters
        #############################################################################

        self.parameters['BatchSize'], self.parameters['TimeStep'], _ = tensorflow.unstack(
            tensorflow.shape(input=self.dataInput, name='DataShape'))

        ###################################################################################################
        # Encoder
        ###################################################################################################

        with tensorflow.variable_scope('Encoder_AE'):
            self.parameters['Encoder_Cell_Forward_AE'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
            self.parameters['Encoder_Cell_Backward_AE'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['Encoder_Output_AE'], self.parameters['Encoder_FinalState_AE'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.parameters['Encoder_Cell_Forward_AE'],
                    cell_bw=self.parameters['Encoder_Cell_Backward_AE'],
                    inputs=self.dataInput, sequence_length=self.seqInput, dtype=tensorflow.float32)

        if self.attention is None:
            self.parameters['Decoder_InitalState_AE'] = []
            for index in range(self.rnnLayers):
                self.parameters['Encoder_Cell_Layer%d_AE' % index] = rnn.LSTMStateTuple(
                    c=tensorflow.concat([self.parameters['Encoder_FinalState_AE'][index][0].c,
                                         self.parameters['Encoder_FinalState_AE'][index][1].c], axis=1),
                    h=tensorflow.concat([self.parameters['Encoder_FinalState_AE'][index][0].h,
                                         self.parameters['Encoder_FinalState_AE'][index][1].h], axis=1))
                self.parameters['Decoder_InitalState_AE'].append(self.parameters['Encoder_Cell_Layer%d_AE' % index])
            self.parameters['Decoder_InitalState_AE'] = tuple(self.parameters['Decoder_InitalState_AE'])
        else:
            self.attentionList = self.attention(dataInput=self.parameters['Encoder_Output_AE'],
                                                scopeName=self.attentionName, hiddenNoduleNumber=2 * self.hiddenNodules,
                                                attentionScope=self.attentionScope, blstmFlag=True)
            self.parameters['Decoder_InitalState_AE'] = []
            for index in range(self.rnnLayers):
                self.parameters['Encoder_Cell_Layer%d_AE' % index] = rnn.LSTMStateTuple(
                    c=self.attentionList['FinalResult'],
                    h=tensorflow.concat(
                        [self.parameters['Encoder_FinalState_AE'][index][0].h,
                         self.parameters['Encoder_FinalState_AE'][index][1].h],
                        axis=1))
                self.parameters['Decoder_InitalState_AE'].append(self.parameters['Encoder_Cell_Layer%d_AE' % index])
            self.parameters['Decoder_InitalState_AE'] = tuple(self.parameters['Decoder_InitalState_AE'])

        #############################################################################
        # Decoder Label Pretreatment
        #############################################################################

        self.parameters['Decoder_Helper_AE'] = seq2seq.TrainingHelper(
            inputs=self.dataInput, sequence_length=self.seqInput, name='Decoder_Helper_AE')
        with tensorflow.variable_scope('Decoder_AE'):
            self.parameters['Decoder_FC_AE'] = Dense(40)
            self.parameters['Decoder_Cell_AE'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules * 2) for _ in range(self.rnnLayers)],
                state_is_tuple=True)

            self.parameters['Decoder_AE'] = seq2seq.BasicDecoder(
                cell=self.parameters['Decoder_Cell_AE'], helper=self.parameters['Decoder_Helper_AE'],
                initial_state=self.parameters['Decoder_InitalState_AE'], output_layer=self.parameters['Decoder_FC_AE'])

            self.parameters['Decoder_Logits_AE'], self.parameters['Decoder_FinalState_AE'], self.parameters[
                'Decoder_FinalSeq_AE'] = seq2seq.dynamic_decode(decoder=self.parameters['Decoder_AE'])

        #############################################################################
        # Losses
        #############################################################################

        self.parameters['Loss_AE'] = tensorflow.losses.absolute_difference(
            labels=self.dataInput, predictions=self.parameters['Decoder_Logits_AE'][0], weights=self.weight)
        self.trainAE = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss_AE'])

        #############################################################################
        # DBLSTM Second BLSTM
        #############################################################################

        with tensorflow.variable_scope('FirstBLSTM'):
            self.parameters['First_Cell_Forward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
            self.parameters['First_Cell_Backward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['First_Output'], self.parameters['First_FinalState'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.parameters['First_Cell_Forward'], cell_bw=self.parameters['First_Cell_Backward'],
                    inputs=self.dataInput, sequence_length=self.seqInput, dtype=tensorflow.float32)

        if self.attention is None:
            self.parameters['First_FinalOutput'] = tensorflow.concat(
                [self.parameters['First_FinalState'][self.rnnLayers - 1][0].h,
                 self.parameters['First_FinalState'][self.rnnLayers - 1][1].h], axis=1)
        else:
            self.firstAttentionList = self.attention(dataInput=self.parameters['First_Output'],
                                                     scopeName=self.attentionName + '_DBLTM',
                                                     hiddenNoduleNumber=2 * self.hiddenNodules,
                                                     attentionScope=self.attentionScope, blstmFlag=True)
            self.parameters['First_FinalOutput'] = self.firstAttentionList['FinalResult']

        if self.concatType == 'Concat':
            self.parameters['First_Concat'] = tensorflow.concat(
                [self.parameters['First_FinalOutput'], self.attentionList['FinalResult']], axis=1, name='First_Concat')
        if self.concatType == 'Plus':
            self.parameters['First_Concat'] = tensorflow.add(self.parameters['First_FinalOutput'],
                                                             self.attentionList['FinalResult'], name='First_Plus')
        if self.concatType == 'Multiply':
            self.parameters['First_Concat'] = tensorflow.multiply(
                self.parameters['First_FinalOutput'], self.attentionList['FinalResult'], name='First_Multiply')

        with tensorflow.variable_scope('SecondBLSTM'):
            self.parameters['Second_Cell_Forward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
            self.parameters['Second_Cell_Backward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['Second_Output'], self.parameters['Second_FinalState'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.parameters['Second_Cell_Forward'], cell_bw=self.parameters['Second_Cell_Backward'],
                    inputs=self.parameters['First_Concat'][tensorflow.newaxis, :, :], dtype=tensorflow.float32)

        if self.secondAttention is None:
            self.parameters['Second_FinalOutput'] = tensorflow.concat(
                [self.parameters['Second_FinalState'][self.rnnLayers - 1][0].h,
                 self.parameters['Second_FinalState'][self.rnnLayers - 1][1].h], axis=1)
        else:
            self.secondAttentionList = self.secondAttention(dataInput=self.parameters['Second_Output'],
                                                            scopeName=self.secondAttentionName,
                                                            hiddenNoduleNumber=2 * self.hiddenNodules,
                                                            attentionScope=self.secondAttentionScope, blstmFlag=True)
            self.parameters['Second_FinalOutput'] = self.secondAttentionList['FinalResult']

        self.parameters['FinalPredict'] = tensorflow.reshape(
            tensor=tensorflow.layers.dense(inputs=self.parameters['Second_FinalOutput'], units=1,
                                           activation=None, name='FinalPredict'), shape=[1])

        if self.lossType == 'MSE':
            self.parameters['Loss'] = tensorflow.losses.mean_squared_error(labels=self.labelInput,
                                                                           predictions=self.parameters['FinalPredict'])
        if self.lossType == 'RMSE':
            self.parameters['Loss'] = tensorflow.sqrt(
                tensorflow.losses.mean_squared_error(labels=self.labelInput,
                                                     predictions=self.parameters['FinalPredict']))
        if self.lossType == 'MAE':
            self.parameters['Loss'] = tensorflow.losses.absolute_difference(labels=self.labelInput,
                                                                            predictions=self.parameters['FinalPredict'])

        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Loss'], var_list=tensorflow.global_variables()[NETWORK_LENGTH[self.attentionName]:])

    def LoadPart(self, loadpath):
        saver = tensorflow.train.Saver(tensorflow.global_variables()[0:NETWORK_LENGTH[self.attentionName]])
        saver.restore(self.session, loadpath)

    def Valid(self):
        trainData, trainLabel, trainSeq = self.treatData, self.treatLabel, self.treatSeq
        # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
        totalLoss = 0.0

        for index in range(len(trainData)):
            batchData, batchLabel, batchSeq = trainData[index], trainLabel[index], trainSeq[index]
            # print(numpy.shape(batchData), numpy.shape(batchLabel), numpy.shape(batchSeq))
            loss = self.session.run(fetches=self.parameters['Loss'],
                                    feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                               self.seqInput: batchSeq})
            print(loss)
            print(numpy.shape(loss))
            exit()
            # print('\rTraining %d/%d Loss = %f' % (index, len(trainData), loss), end='')

    def Train(self, logname):
        with open(logname, 'w') as file:
            trainData, trainLabel, trainSeq = Shuffle_Part3(self.treatData, self.treatLabel, self.treatSeq)
            totalLoss = 0.0
            for index in range(len(trainData)):
                # print(numpy.shape(trainData[index]), numpy.shape(trainLabel[index]), numpy.shape(trainSeq[index]))
                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                           feed_dict={self.dataInput: trainData[index],
                                                      self.labelInput: trainLabel[index],
                                                      self.seqInput: trainSeq[index]})
                file.write(str(loss) + '\n')

                print('\rTraining %d/%d Loss = %f' % (index, len(trainData), loss), end='')
                totalLoss += loss
        return totalLoss
