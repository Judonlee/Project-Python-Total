import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy
import random
from tensorflow.contrib import rnn


def Shuffle(data, label, seqLen):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel, newSeqLen = [], [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
        newSeqLen.append(seqLen[sample])
    return newData, newLabel, newSeqLen


class BLSTM_FinalPooling(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, batchSize=32, learningRate=1e-4,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
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
        self.numClass = numClass
        super(BLSTM_FinalPooling, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                                 learningRate=learningRate, startFlag=startFlag,
                                                 graphRevealFlag=graphRevealFlag,
                                                 graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This Model uses the Final_Pooling to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, self.numClass], name='labelInput')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')

        self.parameters['RNN_Cell_Forward'] = rnn.BasicLSTMCell(num_units=128, name='RNN_Cell_Forward')
        self.parameters['RNN_Cell_Backward'] = rnn.BasicLSTMCell(num_units=128, name='RNN_Cell_Backward')
        (self.parameters['RNN_Output_Forward'], self.parameters['RNN_Output_Backward']), \
        (self.parameters['RNN_FinalState_Forward'], self.parameters['RNN_FinalState_Backward']) = \
            tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['RNN_Cell_Forward'],
                                                    cell_bw=self.parameters['RNN_Cell_Backward'],
                                                    inputs=self.dataInput, sequence_length=self.seqLenInput,
                                                    dtype=tensorflow.float32)

        self.parameters['RNN_Results'] = tensorflow.concat(
            (self.parameters['RNN_FinalState_Forward'].h, self.parameters['RNN_FinalState_Backward'].h), axis=1)

        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Results'],
                                                            units=self.numClass, activation=tensorflow.nn.relu,
                                                            name='Logits')
        self.parameters['PredictProbability'] = tensorflow.nn.softmax(logits=self.parameters['Logits'],
                                                                      name='PredictProbability')
        self.loss = tensorflow.reduce_mean(tensorflow.losses.softmax_cross_entropy(onehot_labels=self.labelInput,
                                                                                   logits=self.parameters['Logits']))
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

            maxlen = numpy.max(trainSeqLen[startPosition:startPosition + self.batchSize])
            for index in range(min(self.batchSize, len(trainData) - startPosition)):
                currentData = numpy.concatenate(
                    (trainData[startPosition + index],
                     numpy.zeros((maxlen - len(trainData[startPosition + index]),
                                  len(trainData[startPosition + index][0])))), axis=0)
                batchTrainData.append(currentData)

            loss, _ = self.session.run(fetches=[self.loss, self.train],
                                       feed_dict={self.dataInput: batchTrainData, self.labelInput: batchTrainLabel,
                                                  self.seqLenInput: batchTrainSeqLen})
            string = '\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss)
            print(string, end='')
            totalLoss += loss
            startPosition += self.batchSize
        return totalLoss

    def Test(self, testData, testLabel, testSeq):
        startPosition = 0

        totalPredict = []

        while startPosition < len(testData):
            string = '\rTesting Batch : %d/%d' % (startPosition, len(testData))
            print(string, end='')
            maxlen = numpy.max(testSeq[startPosition:startPosition + self.batchSize])

            batchTrainData = []
            batchTrainSeqLen = testSeq[startPosition:startPosition + self.batchSize]

            for index in range(min(self.batchSize, len(testData) - startPosition)):
                currentData = numpy.concatenate(
                    (testData[startPosition + index],
                     numpy.zeros((maxlen - len(testData[startPosition + index]),
                                  len(testData[startPosition + index][0])))), axis=0)
                batchTrainData.append(currentData)

            predict = self.session.run(fetches=self.parameters['PredictProbability'],
                                       feed_dict={self.dataInput: batchTrainData,
                                                  self.seqLenInput: batchTrainSeqLen})
            totalPredict.extend(predict)
            startPosition += self.batchSize
        print()
        matrix = numpy.zeros((self.numClass, self.numClass))
        for index in range(len(totalPredict)):
            matrix[numpy.argmax(numpy.array(testLabel[index]))][numpy.argmax(numpy.array(totalPredict[index]))] += 1
        return matrix