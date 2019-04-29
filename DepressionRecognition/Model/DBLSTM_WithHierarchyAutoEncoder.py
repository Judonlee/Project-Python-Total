from DepressionRecognition.Model.DBLSTM import DBLSTM
from __Base.Shuffle import Shuffle_Four, Shuffle_Five
import numpy
import tensorflow
from tensorflow.contrib import rnn


class DBLSTM_WithHierarchyAutoEncoder(DBLSTM):
    def __init__(self, trainData, trainLabel, trainSeq, sentenceLevelInformation=None, speechLevelInformation=None,
                 featureShape=40, firstAttention=None, secondAttention=None, firstAttentionScope=None,
                 secondAttentionScope=None, firstAttentionName=None, secondAttentionName=None, batchSize=32,
                 rnnLayers=2, hiddenNodules=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1, lossType='RMSE'):
        self.sentenceLevelInformation = sentenceLevelInformation
        self.speechLevelInformation = speechLevelInformation
        super(DBLSTM_WithHierarchyAutoEncoder, self).__init__(
            trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, featureShape=featureShape,
            firstAttention=firstAttention, secondAttention=secondAttention, firstAttentionScope=firstAttentionScope,
            secondAttentionScope=secondAttentionScope, firstAttentionName=firstAttentionName,
            secondAttentionName=secondAttentionName, batchSize=batchSize, rnnLayers=rnnLayers,
            hiddenNodules=hiddenNodules, learningRate=learningRate, startFlag=startFlag,
            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate, lossType=lossType)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=None, name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int64, shape=None, name='seqInput')
        self.sentenceLevelInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[None, 4 * self.hiddenNodules], name='sentenceLevelInput')
        self.speechLevelInput = tensorflow.placeholder(
            dtype=tensorflow.float32, shape=[1, 2 * self.hiddenNodules], name='speechLevelInput')

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

        if self.sentenceLevelInformation is not None:
            self.parameters['First_FinalOutput'] = tensorflow.concat(
                [self.parameters['First_FinalOutput'], self.sentenceLevelInput], axis=1)

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

        if self.speechLevelInformation is not None:
            self.parameters['Second_FinalOutput'] = tensorflow.concat(
                [self.parameters['Second_FinalOutput'], self.speechLevelInput], axis=1)

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

        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Valid(self):
        trainData, trainLabel, trainSeq, trainSentence = self.data, self.label, self.seq, self.sentenceLevelInformation

        result = self.session.run(fetches=self.parameters['Loss'],
                                  feed_dict={self.dataInput: trainData[0], self.labelInput: trainLabel[0],
                                             self.seqInput: trainSeq[0], self.sentenceLevelInput: trainSentence[0]})
        print(result)
        print(numpy.shape(result))

    def Train(self, logName):
        if self.sentenceLevelInformation is None:
            treatSentence = numpy.ones([len(self.data), 1])
        else:
            treatSentence = self.sentenceLevelInformation

        if self.speechLevelInformation is None:
            treatSpeech = numpy.ones([len(self.data), 1])
        else:
            treatSpeech = self.speechLevelInformation

        trainData, trainLabel, trainSeq, trainSentenceLevel, trainSpeechlevel = \
            Shuffle_Five(self.data, self.label, self.seq, treatSentence, treatSpeech)
        totalLoss = 0
        with open(logName, 'w') as file:
            for index in range(len(trainData)):

                if self.sentenceLevelInformation is None:
                    batchSentence = numpy.ones([len(trainData[index]), 2 * self.hiddenNodules])
                else:
                    batchSentence = trainSentenceLevel[index]

                if self.speechLevelInformation is None:
                    batchSpeech = numpy.ones([1, 2 * self.hiddenNodules])
                else:
                    batchSpeech = trainSpeechlevel[index]

                loss, _ = self.session.run(
                    fetches=[self.parameters['Loss'], self.train],
                    feed_dict={self.dataInput: trainData[index], self.labelInput: trainLabel[index],
                               self.seqInput: trainSeq[index], self.sentenceLevelInput: batchSentence,
                               self.speechLevelInput: batchSpeech})
                totalLoss += loss
                file.write(str(loss) + '\n')
                print('\rTraining %d/%d Loss = %f' % (index, len(trainData), loss), end='')
        return totalLoss

    def Test(self, logName, testData, testLabel, testSeq, testSentence=None, testSpeech=None):
        with open(logName, 'w') as file:
            for index in range(len(testData)):
                print('\rTreating %d/%d' % (index, len(testData)), end='')
                batchData, batchLabel, batchSeq, batchSentence, batchSpeech = \
                    testData[index], testLabel[index], testSeq[index], numpy.zeros(
                        [len(testData[index]), 2 * self.hiddenNodules]), numpy.zeros([1, 2 * self.hiddenNodules])
                if testSentence is not None: batchSentence = testSentence[index]
                if testSpeech is not None: batchSpeech = testSpeech[index]

                predict = self.session.run(
                    fetches=self.parameters['FinalPredict'],
                    feed_dict={self.dataInput: batchData, self.seqInput: batchSeq,
                               self.sentenceLevelInput: batchSentence, self.speechLevelInput: batchSpeech})
                file.write(str(batchLabel[0]) + ',' + str(predict[0]) + '\n')
