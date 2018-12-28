import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy
from __Base.Shuffle import Shuffle_Train


class DNN_WithDropOut(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, featureShape, numClass, hiddenNodules=128, batchSize=32,
                 learningRate=1e-3, startFlag=True, graphRevealFlag=False, graphPath='logs/', occupyRate=-1):
        '''
        :param trainLabel:      In this case, trainLabel is the targets.
        :param featureShape:    designate how many features in one vector.
        :param hiddenNodules:   designite the number of hidden nodules.
        :param numClass:        designite the number of classes
        '''

        self.featureShape = featureShape
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        super(DNN_WithDropOut, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                              learningRate=learningRate, startFlag=startFlag,
                                              graphRevealFlag=graphRevealFlag,
                                              graphPath=graphPath, occupyRate=occupyRate)
        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, self.numClass], name='labelInput')
        self.keepProbability = tensorflow.placeholder(dtype=tensorflow.float32, name='keepProbability')

        self.parameters['Layer1st'] = tensorflow.layers.dense(inputs=self.dataInput, units=self.hiddenNodules,
                                                              activation=tensorflow.nn.relu, name='Layer1st')
        self.parameters['Layer1st_DropOut'] = tensorflow.nn.dropout(
            x=self.parameters['Layer1st'], keep_prob=self.keepProbability, name='Layer1st_DropOut')

        #################################################################################################

        self.parameters['Layer2nd'] = tensorflow.layers.dense(inputs=self.parameters['Layer1st_DropOut'],
                                                              units=self.hiddenNodules, activation=tensorflow.nn.relu,
                                                              name='Layer2nd')
        self.parameters['Layer2nd_DropOut'] = tensorflow.nn.dropout(
            x=self.parameters['Layer2nd'], keep_prob=self.keepProbability, name='Layer2nd_DropOut')

        #################################################################################################

        self.parameters['Layer3rd'] = tensorflow.layers.dense(inputs=self.parameters['Layer2nd_DropOut'],
                                                              units=self.hiddenNodules, activation=tensorflow.nn.relu,
                                                              name='Layer3rd')
        self.parameters['Layer3rd_DropOut'] = tensorflow.nn.dropout(
            x=self.parameters['Layer3rd'], keep_prob=self.keepProbability, name='Layer3rd_DropOut')

        #################################################################################################

        self.parameters['Layer4th'] = tensorflow.layers.dense(inputs=self.parameters['Layer3rd_DropOut'],
                                                              units=self.numClass, activation=tensorflow.nn.relu,
                                                              name='Layer4th')

        self.loss = tensorflow.losses.softmax_cross_entropy(onehot_labels=self.labelInput,
                                                            logits=self.parameters['Layer4th'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.loss)

    def Train(self):
        trainData, trainLabel = Shuffle_Train(data=self.data, label=self.label)

        startPosition = 0
        totalLoss = 0.0
        # print(numpy.shape(trainData),numpy.shape(trainLabel))
        # print()
        while startPosition < len(trainData):
            # print('HERE')
            batchData = trainData[startPosition:startPosition + self.batchSize]
            batchLabel = trainLabel[startPosition:startPosition + self.batchSize]

            loss, _ = self.session.run(fetches=[self.loss, self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                  self.keepProbability: 0.5})
            totalLoss += loss
            startPosition += self.batchSize
            print('\rTraining\t:\t%d\t/\t%d\t\tLoss = %f' % (startPosition, len(trainData), loss), end='')
        return totalLoss

    def Test(self, testData, testLabel):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            batchData = testData[startPosition:startPosition + self.batchSize]

            predict = self.session.run(fetches=self.parameters['Layer4th'],
                                       feed_dict={self.dataInput: batchData, self.keepProbability: 1})
            totalPredict.extend(predict)
            startPosition += self.batchSize
            print('\rTesting\t:\t%d\t/\t%d\t' % (startPosition, len(testData)), end='')

        matrix = numpy.zeros((self.numClass, self.numClass))
        for index in range(len(testLabel)):
            matrix[numpy.argmax(numpy.array(testLabel[index]))][numpy.argmax(numpy.array(totalPredict[index]))] += 1

        print(matrix)
        return matrix
