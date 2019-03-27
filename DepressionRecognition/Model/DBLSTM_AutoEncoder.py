import tensorflow
import numpy
from DepressionRecognition.Model.AutoEncoder import AutoEncoder
from __Base.Shuffle import Shuffle_Train
from tensorflow.contrib import rnn
from tensorflow.contrib import seq2seq
from tensorflow.python.layers.core import Dense

NETWORK_LENGTH = {'None': 44, 'SA': 50, 'LA': 50, 'MA': 56, 'MCA': 50}


class DBLSTM_AutoEncoder(AutoEncoder):
    def __init__(self, data, label, seq, lossType='MAE', weight=10, attention=None, attentionName=None,
                 attentionScope=None, secondAttention=None, secondAttentionName=None, secondAttentionScope=None,
                 batchSize=32, maxLen=1000, rnnLayers=2, hiddenNodules=128, learningRate=1E-3, startFlag=True,
                 graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.treatLabel = label
        self.lossType = lossType

        self.secondAttention = secondAttention
        self.secondAttentionName = secondAttentionName
        self.secondAttentionScope = secondAttentionScope

        super(DBLSTM_AutoEncoder, self).__init__(
            data=data, seq=seq, weight=weight, attention=attention, attentionName=attentionName,
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
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss_AE'])

        #############################################################################
        # DBLSTM Second BLSTM
        #############################################################################

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
                    inputs=self.attentionList['FinalResult'][tensorflow.newaxis, :, :],
                    dtype=tensorflow.float32)

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
                                                                           predictions=self.parameters['FinalPredict'],
                                                                           weights=24)
        if self.lossType == 'RMSE':
            self.parameters['Loss'] = tensorflow.sqrt(
                tensorflow.losses.mean_squared_error(labels=self.labelInput,
                                                     predictions=self.parameters['FinalPredict'], weights=24))
        if self.lossType == 'MAE':
            self.parameters['Loss'] = tensorflow.losses.absolute_difference(labels=self.labelInput,
                                                                            predictions=self.parameters['FinalPredict'],
                                                                            weights=24)

        self.train_DR_Part = tensorflow.train.AdamOptimizer(
            learning_rate=learningRate).minimize(self.parameters['Loss'],
                                                 var_list=tensorflow.global_variables()[
                                                          NETWORK_LENGTH[self.attentionName]:])
        self.train_DR_Total = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Loss'])

    def AutoEncoderTrain(self, logname):
        data, seq = Shuffle_Train(data=self.treatData, label=self.treatSeq)

        with open(logname, 'w') as file:
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
                            [data[index],
                             numpy.zeros([maxSeq - numpy.shape(data[index])[0], numpy.shape(data[index])[1]])],
                            axis=0)
                    # print(numpy.shape(currentData))
                    batchData.append(currentData)

                # print(startPosition, numpy.shape(batchData), numpy.shape(batchSeq), maxSeq)
                # print(batchSeq, '\n')

                loss, _ = self.session.run(fetches=[self.parameters['Loss_AE'], self.train],
                                           feed_dict={self.dataInput: batchData, self.seqInput: batchSeq})
                file.write(str(loss) + '\n')
                totalLoss += loss
                print('\rBatch %d/%d Loss = %f' % (startPosition, len(data), loss), end='')

                # exit()

                startPosition += self.batchSize
        return totalLoss

    def LoadPart(self, loadpath):
        saver = tensorflow.train.Saver(tensorflow.global_variables()[0:NETWORK_LENGTH[self.attentionName]])
        saver.restore(self.session, loadpath)

    def TrainPart(self, logname):
        with open(logname, 'w') as file:
            trainData, trainLabel, trainSeq = self.treatData, self.treatLabel, self.treatSeq
            # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(trainSeq))
            totalLoss = 0.0

            for index in range(len(trainData)):
                batchData, batchLabel, batchSeq = trainData[index], trainLabel[index], trainSeq[index]
                # print(numpy.shape(batchData), numpy.shape(batchLabel), numpy.shape(batchSeq))
                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train_DR_Part],
                                           feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                      self.seqInput: batchSeq})
                file.write(str(loss) + '\n')
                totalLoss += loss
                print('\rTraining %d/%d Loss = %f' % (index, len(trainData), loss), end='')
        return totalLoss

    def Test(self, logname):
        pass
