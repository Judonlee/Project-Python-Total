from __Base.BaseClass import NeuralNetwork_Base
from DepressionRecognition.Tools import Shuffle_Part3
import numpy
import tensorflow
from tensorflow.contrib import rnn


class DBLSTM(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeq, featureShape=40, firstAttention=None, secondAttention=None,
                 firstAttentionScope=None, secondAttentionScope=None, firstAttentionName=None, secondAttentionName=None,
                 batchSize=32, rnnLayers=2, hiddenNodules=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        self.seq = trainSeq
        self.featureShape = featureShape
        self.firstAttention, self.secondAttention = firstAttention, secondAttention

        self.firstAttentionScope, self.secondAttentionScope = firstAttentionScope, secondAttentionScope
        self.firstAttentionName, self.secondAttentionName = firstAttentionName, secondAttentionName

        self.rnnLayers, self.hiddenNodules = rnnLayers, hiddenNodules

        super(DBLSTM, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                     learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                     graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=None, name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int64, shape=None, name='seqInput')

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

        ##########################################################################

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
        self.parameters['Loss'] = tensorflow.losses.mean_squared_error(labels=self.labelInput,
                                                                       predictions=self.parameters['FinalPredict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self, logName):
        trainData, trainLabel, trainSeq = Shuffle_Part3(data=self.data, label=self.label, seq=self.seq)
        totalLoss = 0
        with open(logName, 'w') as file:
            for index in range(len(trainData)):
                loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                           feed_dict={self.dataInput: trainData[index],
                                                      self.labelInput: trainLabel[index],
                                                      self.seqInput: trainSeq[index]})
                totalLoss += loss
                file.write(str(loss) + '\n')
                print('\rTraining %d/%d Loss = %f' % (index, len(trainData), loss), end='')
        return totalLoss

    def Test(self, testData, testLabel, testSeq, logName):
        with open(logName, 'w') as file:
            for index in range(len(testData)):
                print('\rTesting %d/%d' % (index, len(testData)), end='')
                predict = self.session.run(fetches=self.parameters['FinalPredict'],
                                           feed_dict={self.dataInput: testData[index], self.seqInput: testSeq[index]})
                file.write(str(testLabel[index]) + ',' + str(predict[0]) + '\n')

    def Valid(self):
        trainData, trainLabel, trainSeq = self.data, self.label, self.seq

        result = self.session.run(fetches=self.parameters['FinalPredict'],
                                  feed_dict={self.dataInput: trainData[0], self.labelInput: trainLabel[0],
                                             self.seqInput: trainSeq[0]})
        print(result)
        print(numpy.shape(result))
