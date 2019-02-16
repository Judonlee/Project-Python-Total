import numpy
import tensorflow
from tensorflow.contrib import rnn
from __Base.BaseClass import NeuralNetwork_Base

FEATURE_SHAPE = 6


class BLSTM(NeuralNetwork_Base):
    def __init__(self, trainData, considerScope=7, batchSize=32, learningRate=1E-3, hiddenNodules=128,
                 rnnLayers=2, startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.considerScope = considerScope
        self.hiddenNodules = hiddenNodules
        self.rnnLayers = rnnLayers
        super(BLSTM, self).__init__(trainData=trainData, trainLabel=None, batchSize=batchSize,
                                    learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                    graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32,
                                                shape=[None, self.considerScope, FEATURE_SHAPE], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, FEATURE_SHAPE],
                                                 name='labelInput')

        ##########################################################################

        with tensorflow.variable_scope('BLSTM'):
            self.parameters['FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
            self.parameters['BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['Output'], self.parameters['FinalState'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['FW_Cell'],
                                                        cell_bw=self.parameters['BW_Cell'], inputs=self.dataInput,
                                                        dtype=tensorflow.float32)

        self.parameters['FinalOutput'] = tensorflow.concat(
            [self.parameters['FinalState'][-1][0].h, self.parameters['FinalState'][-1][1].h], axis=1,
            name='FinalOutput')
        self.parameters['Predict'] = tensorflow.layers.dense(inputs=self.parameters['FinalOutput'], units=FEATURE_SHAPE,
                                                             activation=None, name='Predict')

        self.parameters['Loss'] = tensorflow.losses.mean_squared_error(labels=self.labelInput,
                                                                       predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        trainData, trainLabel = [], []
        for index in range(len(self.data) - self.considerScope):
            trainData.append(self.data[index:index + self.considerScope])
            trainLabel.append(self.data[index + self.considerScope])

        startPosition = 0
        totalLoss = 0.0
        while startPosition < numpy.shape(trainData)[0]:
            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train], feed_dict={
                self.dataInput: trainData[startPosition:startPosition + self.batchSize],
                self.labelInput: trainLabel[startPosition:startPosition + self.batchSize]})
            print('\rTraining %d/%d' % (startPosition, numpy.shape(trainData)[0]), end='')
            totalLoss += loss
            startPosition += self.batchSize
        return totalLoss

    def SingleTest(self, testData):
        result = self.session.run(fetches=self.parameters['Predict'], feed_dict={self.dataInput: testData})
        print(result)
        return result
