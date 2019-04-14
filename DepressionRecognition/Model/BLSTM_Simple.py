import tensorflow
import numpy
from __Base.BaseClass import NeuralNetwork_Base
from __Base.Shuffle import Shuffle
from tensorflow.contrib import rnn


class BLSTM(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeq, lossType='MAE', attention=None, attentionName='',
                 attentionScope=0, rnnLayers=2, hiddenNoduleNumber=128, batchSize=8, learningRate=1E-3, startFlag=True,
                 graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.seq = trainSeq
        self.lossType = lossType

        self.attention = attention
        self.attentionName = attentionName
        self.attentionScope = attentionScope

        self.rnnLayers = rnnLayers
        self.hiddenNoduleNumber = hiddenNoduleNumber

        super(BLSTM, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                    learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                    graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, 256], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None], name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int64, shape=[None], name='seqInput')

        self.parameters['First_FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNoduleNumber) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameters['First_BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNoduleNumber) for _ in range(self.rnnLayers)], state_is_tuple=True)

        self.parameters['First_Output'], self.parameters['First_FinalState'] = \
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=self.parameters['First_FW_Cell'], cell_bw=self.parameters['First_BW_Cell'],
                inputs=self.dataInput, sequence_length=self.seqInput, dtype=tensorflow.float32)

        if self.attention is None:
            self.parameters['First_FinalOutput'] = tensorflow.concat(
                [self.parameters['First_FinalState'][self.rnnLayers - 1][0].h,
                 self.parameters['First_FinalState'][self.rnnLayers - 1][1].h], axis=1)
        else:
            self.firstAttentionList = self.attention(
                dataInput=self.parameters['First_Output'], scopeName=self.attentionName,
                hiddenNoduleNumber=2 * self.hiddenNoduleNumber, attentionScope=self.attentionScope, blstmFlag=True)
            self.parameters['First_FinalOutput'] = self.firstAttentionList['FinalResult']

        self.parameters['FinalPredict'] = tensorflow.layers.dense(inputs=self.parameters['First_FinalOutput'], units=1,
                                                                  activation=None, name='FinalPredict')

        if self.lossType == 'RMSE':
            self.parameters['Loss'] = tensorflow.sqrt(
                tensorflow.losses.mean_squared_error(
                    labels=self.labelInput, predictions=tensorflow.reshape(self.parameters['FinalPredict'], [-1])))
        if self.lossType == 'MAE':
            self.parameters['Loss'] = tensorflow.losses.absolute_difference(
                labels=self.labelInput, predictions=tensorflow.reshape(self.parameters['FinalPredict'], [-1]))

        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        startPosition = 0
        trainData, trainLabel, trainSeq = Shuffle(self.data, self.label, self.seq)
        # trainData, trainLabel, trainSeq = self.data, self.label, self.seq
        totalLoss = 0.0
        while startPosition < len(trainData):
            batchData, batchLabel, batchSeq = [], trainLabel[startPosition:startPosition + self.batchSize], \
                                              trainSeq[startPosition:startPosition + self.batchSize]

            for index in range(min(self.batchSize, len(trainData) - startPosition)):
                currentData = numpy.concatenate([trainData[startPosition + index], numpy.zeros(
                    [max(batchSeq) - numpy.shape(trainData[startPosition + index])[0],
                     numpy.shape(trainData[startPosition + index])[1]])], axis=0)
                # print(numpy.shape(currentData))
                batchData.append(currentData)

            # print(numpy.shape(batchData), numpy.shape(batchLabel), numpy.shape(batchSeq))

            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                  self.seqInput: batchSeq})
            totalLoss += loss
            print('\rTraining %d/%d Loss = %f' % (startPosition, len(trainData), loss), end='')
            startPosition += self.batchSize
        return totalLoss

    def Test(self, logname, testData, testLabel, testSeq):
        startPosition = 0

        with open(logname, 'w') as file:
            totalPredict = []
            while startPosition < len(testData):
                batchData, batchLabel, batchSeq = [], testLabel[startPosition:startPosition + self.batchSize], \
                                                  testSeq[startPosition:startPosition + self.batchSize]

                for index in range(min(self.batchSize, len(testData) - startPosition)):
                    currentData = numpy.concatenate([testData[startPosition + index], numpy.zeros(
                        [max(batchSeq) - numpy.shape(testData[startPosition + index])[0],
                         numpy.shape(testData[startPosition + index])[1]])], axis=0)
                    # print(numpy.shape(currentData))
                    batchData.append(currentData)

                # print(numpy.shape(batchData), numpy.shape(batchLabel), numpy.shape(batchSeq))

                predict = self.session.run(fetches=self.parameters['FinalPredict'],
                                           feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                      self.seqInput: batchSeq})
                totalPredict.extend(predict)
                startPosition += self.batchSize

            for index in range(len(totalPredict)):
                file.write(str(testLabel[index]) + ',' + str(totalPredict[index][0]) + '\n')
