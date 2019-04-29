import tensorflow
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import seq2seq
import numpy
import os
from __Base.BaseClass import NeuralNetwork_Base
from __Base.Shuffle import Shuffle


class HierarchyAutoEncoder(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeq, lossType='frame', firstAttention=None, firstAttentionName='',
                 firstAttentionScope=0, secondAttention=None, secondAttentionName='', secondAttentionScope=0,
                 rnnLayers=2, hiddenNoduleNumber=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.seq = trainSeq
        self.lossType = lossType

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
        else:
            self.attentionListSecond = self.secondAttention(dataInput=self.parameters['Second_Output'],
                                                            scopeName=self.secondAttentionName,
                                                            hiddenNoduleNumber=2 * self.hiddenNodules,
                                                            attentionScope=self.secondAttentionScope, blstmFlag=True)
            self.parameters['Decoder_InitalState'] = []
            self.attentionListSecond['FinalResult'].set_shape([1, 2 * self.hiddenNodules])
            self.parameters['FinalResult'] = self.attentionListSecond['FinalResult']
            for index in range(self.rnnLayers):
                self.parameters['Encoder_Cell_Layer%d' % index] = rnn.LSTMStateTuple(
                    c=self.attentionListSecond['FinalResult'],
                    h=tensorflow.concat(
                        [self.parameters['Second_FinalState'][index][0].h,
                         self.parameters['Second_FinalState'][index][1].h],
                        axis=1))
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

        with tensorflow.variable_scope('LossScope'):
            if self.lossType == 'frame':
                self.parameters['Loss'] = tensorflow.losses.absolute_difference(
                    labels=self.dataInput, predictions=self.parameters['Decoder_Logits_Second'][0], weights=100)
            if self.lossType == 'sentence':
                self.parameters['Loss'] = tensorflow.losses.absolute_difference(
                    labels=self.parameters['First_FinalOutput'][tensorflow.newaxis, :, :],
                    predictions=self.parameters['Decoder_Logits_First'][0],
                    weights=100)
            self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self, logname):
        with open(logname, 'w') as file:
            trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seq)
            totalLoss = 0.0
            # trainData, trainLabel, trainSeq = self.data, self.label, self.seq
            for index in range(len(trainData)):
                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                           feed_dict={self.dataInput: trainData[index],
                                                      self.labelInput: trainLabel[index],
                                                      self.seqInput: trainSeq[index]})
                totalLoss += loss
                file.write(str(loss) + '\n')
                print('\rTraining %d/%d Loss = %f' % (index, len(trainData), loss), end='')
        return totalLoss

    def Valid(self):
        trainData, trainLabel, trainSeq = self.data, self.label, self.seq
        for index in range(len(trainData)):
            loss = self.session.run(fetches=self.parameters['Decoder_Logits_First'],
                                    feed_dict={self.dataInput: trainData[index], self.labelInput: trainLabel[index],
                                               self.seqInput: trainSeq[index]})
            print(loss)
            # print(numpy.shape(loss))
            exit()

    def TestOut(self, logname, treatData, treatSeq, treatname):
        with open(logname, 'w') as file:
            for index in range(len(treatData)):
                print('\rTreating %d/%d' % (index, len(treatData)), end='')
                result = self.session.run(fetches=self.parameters[treatname],
                                          feed_dict={self.dataInput: treatData[index], self.seqInput: treatSeq[index]})
                print(numpy.shape(result))
                result = numpy.reshape(result, [-1, 1024])
                for writeIndex in range(512, 1024):
                    if writeIndex != 512: file.write(',')
                    file.write(str(result[0][writeIndex]))
                file.write('\n')
        print('\nTreat Completed')

    def TestOutMedia(self, savepath, treatData, treatSeq, treatname):
        os.makedirs(savepath)
        for index in range(len(treatData)):
            result = self.session.run(fetches=self.parameters[treatname],
                                      feed_dict={self.dataInput: treatData[index], self.seqInput: treatSeq[index]})
            print(numpy.shape(result))
            result = numpy.reshape(result, [-1, 1024])
            print('\rTreating %d/%d' % (index, len(treatData)) + str(numpy.shape(result)), end='')

            with open(savepath + '/%04d.csv' % index, 'w') as file:
                for indexX in range(numpy.shape(result)[0]):
                    for indexY in range(512, 1024):
                        if indexY != 512: file.write(',')
                        file.write(str(result[indexX][indexY]))
                    file.write('\n')

    def TestOutHuge(self, savepath, treatData, treatSeq, treatname):
        os.makedirs(savepath)
        for index in range(len(treatData)):
            result = self.session.run(fetches=self.parameters[treatname],
                                      feed_dict={self.dataInput: treatData[index], self.seqInput: treatSeq[index]})
            # result = numpy.concatenate([result[0], result[1]], axis=2)
            print('\rTreating %d/%d' % (index, len(treatData)) + str(numpy.shape(result)), end='')
            numpy.save(savepath + '%04d.npy' % index, result)
