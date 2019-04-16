import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
from __Base.Shuffle import Shuffle_Train
import numpy


class DNN(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, batchSize=8, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        super(DNN, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                  learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                  graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 256], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 1], name='labelInput')

        self.parameters['FirstLayer'] = tensorflow.layers.dense(
            inputs=self.dataInput, units=256, activation=tensorflow.nn.relu, name='FirstLayer')
        self.parameters['SecondLayer'] = tensorflow.layers.dense(
            inputs=self.parameters['FirstLayer'], units=256, activation=tensorflow.nn.relu, name='SecondLayer')
        self.parameters['ThirdLayer'] = tensorflow.layers.dense(
            inputs=self.parameters['SecondLayer'], units=256, activation=tensorflow.nn.relu, name='ThirdLayer')
        self.parameters['Predict'] = tensorflow.layers.dense(
            inputs=self.parameters['ThirdLayer'], units=1, activation=None, name='Predict')
        self.parameters['Loss'] = tensorflow.losses.absolute_difference(
            labels=self.labelInput, predictions=self.parameters['Predict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        trainData, trainLabel = Shuffle_Train(data=self.data, label=self.label)
        startPosition = 0
        totalLoss = 0.0
        while startPosition < len(trainData):
            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train], feed_dict={
                self.dataInput: trainData[startPosition:startPosition + self.batchSize],
                self.labelInput: numpy.reshape(trainLabel[startPosition:startPosition + self.batchSize], [-1, 1])})
            print('\rTrain %d/%d Loss = %f' % (startPosition, len(trainData), loss), end='')
            totalLoss += loss
            startPosition += self.batchSize
        return totalLoss

    def Test(self, logname, testData, testLabel):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            predict = self.session.run(fetches=self.parameters['Predict'], feed_dict={
                self.dataInput: testData[startPosition:startPosition + self.batchSize]})
            totalPredict.extend(predict)
            startPosition += self.batchSize

        with open(logname, 'w') as file:
            for index in range(len(testLabel)):
                file.write(str(testLabel[index]) + ',' + str(totalPredict[index][0]) + '\n')
