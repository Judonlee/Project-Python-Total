import tensorflow
from CTC_Project.Module.BaseClass import NeuralNetwork_Base
import numpy
import random


def Shuffle(data, label, seqLen):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel, newSeqLen = [], [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
        newSeqLen.append(seqLen[sample])
    return newData, newLabel, newSeqLen


class LSTM_FinalPooling(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, hiddenNodules=128, rnnLayers=1,
                 batchSize=32, learningRate=0.0001, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        '''
        :param trainLabel:      In this case, trainLabel is the targets.
        :param trainSeqLength:  In this case, the trainSeqLength are needed which is the length of each cases.
        :param featureShape:    designate how many features in one vector.
        :param hiddenNodules:   designite the number of hidden nodules.
        :param rnnLayers:       designate the number of rnn layers.
        :param numClass:        designite the number of classes
        '''

        self.featureShape = featureShape
        self.seqLen = trainSeqLength
        self.hiddenNodules = hiddenNodules
        self.rnnLayer = rnnLayers
        super(LSTM_FinalPooling, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                                learningRate=learningRate, startFlag=startFlag,
                                                graphRevealFlag=graphRevealFlag,
                                                graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This Model uses the Final_Pooling to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, 1], name='labelInput')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')
        self.keepProbability = tensorflow.placeholder(dtype=tensorflow.float32, shape=None, name='keepProbability')

        self.rnnCell = []
        for layers in range(self.rnnLayer):
            self.parameters['RNN_Cell_Layer_' + str(layers)] = tensorflow.contrib.rnn.LSTMCell(self.hiddenNodules)
            self.rnnCell.append(self.parameters['RNN_Cell_Layer_' + str(layers)])

        self.parameters['Stack'] = tensorflow.contrib.rnn.MultiRNNCell(self.rnnCell)

        self.parameters['RNN_Outputs'], self.parameters['RNN_FinalState'] = \
            tensorflow.nn.dynamic_rnn(cell=self.parameters['Stack'], inputs=self.dataInput,
                                      sequence_length=self.seqLenInput, dtype=tensorflow.float32)

        self.parameters['RNN_Results'] = tensorflow.reshape(tensor=self.parameters['RNN_FinalState'][0][1],
                                                            shape=[-1, self.hiddenNodules], name='RNN_Results')
        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Results'],
                                                            units=1, name='Logits')
        self.loss = tensorflow.reduce_mean(
            tensorflow.losses.mean_squared_error(labels=self.labelInput, predictions=self.parameters['Logits']))
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.loss)

    def Train(self):
        print()
        totalLoss = 0
        trainData, trainLabel, trainSeqLen = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)

        startPosition = 0
        while startPosition < len(trainData):
            batchTrainData = []
            batchTrainLabel = trainLabel[startPosition:startPosition + self.batchSize]
            batchTrainSeqLen = trainSeqLen[startPosition:startPosition + self.batchSize]

            maxlen = 0
            for index in range(min(self.batchSize, len(trainData) - startPosition)):
                if len(trainData[startPosition + index]) > maxlen:
                    maxlen = len(trainData[startPosition + index])
            for index in range(min(self.batchSize, len(trainData) - startPosition)):
                currentData = numpy.concatenate(
                    (trainData[startPosition + index],
                     numpy.zeros((maxlen - len(trainData[startPosition + index]),
                                  len(trainData[startPosition + index][0])))),
                    axis=0)
                batchTrainData.append(currentData)

            loss, _ = self.session.run(fetches=[self.loss, self.train],
                                       feed_dict={self.dataInput: batchTrainData,
                                                  self.labelInput: numpy.reshape(batchTrainLabel, [-1, 1]),
                                                  self.seqLenInput: batchTrainSeqLen})

            string = '\rBatch :' + str(int(startPosition / self.batchSize)) + '/' + str(
                int(len(trainData) / self.batchSize)) + '\t' + str(numpy.shape(batchTrainData)) + '\t' + str(
                numpy.shape(batchTrainLabel)) + '\t' + str(numpy.shape(batchTrainSeqLen)) + '\tLoss : ' + str(loss)
            print(string, end='')
            totalLoss += loss
            startPosition += self.batchSize
        return totalLoss

    def MAE_Calculation(self, labels, predict):
        counter = 0
        for index in range(len(labels)):
            counter += numpy.abs(labels[index] - predict[index])
        counter /= len(labels)
        return counter

    def RMSE_Calculation(self, labels, predict):
        counter = 0
        for index in range(len(labels)):
            counter += (labels[index] - predict[index]) * (labels[index] - predict[index])
        counter /= len(labels)
        counter = numpy.sqrt(counter)
        return counter

    def Test(self, testData, testLabel, testSequence, savename='None'):
        totalPredict = []
        startPosition = 0
        while startPosition < len(testData):
            batchTestData = []

            maxlen = 0
            for index in range(min(self.batchSize, len(testData) - startPosition)):
                if maxlen < len(testData[startPosition + index]): maxlen = len(testData[startPosition + index])

            for index in range(min(self.batchSize, len(testData) - startPosition)):
                currentData = numpy.concatenate(
                    (testData[startPosition + index], numpy.zeros(
                        (maxlen - len(testData[startPosition + index]), len(testData[startPosition + index][0])))),
                    axis=0)
                batchTestData.append(currentData)

            predict = self.session.run(fetches=self.parameters['Logits'], feed_dict={
                self.dataInput: batchTestData,
                self.seqLenInput: testSequence[startPosition:startPosition + self.batchSize]})
            totalPredict.extend(predict)
            startPosition += self.batchSize

        MAE, RMSE = self.MAE_Calculation(labels=testLabel, predict=totalPredict), \
                    self.RMSE_Calculation(labels=testLabel, predict=totalPredict)

        if savename != 'None':
            file = open(savename, 'w')
            file.write(str(MAE) + ',' + str(RMSE))
            file.close()
        return MAE, RMSE
