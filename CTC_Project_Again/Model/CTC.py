import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
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


class CTC(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, rnnLayers=1,
                 batchSize=32, learningRate=1e-4, startFlag=True, graphRevealFlag=True, graphPath='logs/',
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
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        self.rnnLayer = rnnLayers
        super(CTC, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                  learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                  graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This Model uses the Final_Pooling to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.sparse_placeholder(dtype=tensorflow.int32, shape=None, name='labelInput')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]
        self.parameters['TimeStep'] = tensorflow.shape(input=self.dataInput, name='TimeStep')[1]

        ###################################################################################################
        # RNN Start
        ###################################################################################################

        self.rnnCell = []
        for layers in range(self.rnnLayer):
            self.parameters['RNN_Cell_Layer_' + str(layers)] = tensorflow.contrib.rnn.LSTMCell(self.hiddenNodules)
            self.rnnCell.append(self.parameters['RNN_Cell_Layer_' + str(layers)])
        self.parameters['Stack'] = tensorflow.contrib.rnn.MultiRNNCell(self.rnnCell)
        self.parameters['RNN_Outputs'], self.parameters['RNN_FinalState'] = \
            tensorflow.nn.dynamic_rnn(cell=self.parameters['Stack'], inputs=self.dataInput,
                                      sequence_length=self.seqLenInput, dtype=tensorflow.float32)
        self.parameters['RNN_Outputs_Reshape'] = tensorflow.reshape(tensor=self.parameters['RNN_Outputs'],
                                                                    shape=[-1, self.hiddenNodules])

        ###################################################################################################
        # CTC Start
        ###################################################################################################

        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Outputs_Reshape'],
                                                            units=self.numClass, name='logits')
        self.parameters['Logits_Reshape'] = tensorflow.reshape(tensor=self.parameters['Logits'],
                                                               shape=[self.parameters['BatchSize'], -1, self.numClass],
                                                               name='Logits_Reshape')
        self.parameters['Logits_TimeMajor'] = tensorflow.transpose(a=self.parameters['Logits_Reshape'], perm=(1, 0, 2),
                                                                   name='Logits_TimeMajor')
        self.parameters['Loss'] = tensorflow.nn.ctc_loss(labels=self.labelInput,
                                                         inputs=self.parameters['Logits_TimeMajor'],
                                                         sequence_length=self.seqLenInput)
        self.parameters['Cost'] = tensorflow.reduce_mean(self.parameters['Loss'], name='Cost')
        self.train = tensorflow.train.MomentumOptimizer(learning_rate=learningRate, momentum=0.9). \
            minimize(self.parameters['Cost'])

        self.decode, self.logProbability = tensorflow.nn.ctc_beam_search_decoder(
            inputs=self.parameters['Logits_TimeMajor'],
            sequence_length=self.seqLenInput,
            merge_repeated=False)
        self.decodeDense = tensorflow.sparse_tensor_to_dense(sp_input=self.decode[0])

    def Train(self):
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)

        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchData = []
            batachSeq = trainSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(trainSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                currentData = numpy.concatenate(
                    (trainData[index], numpy.zeros((maxLen - len(trainData[index]), len(trainData[index][0])))), axis=0)
                batchData.append(currentData)

            indices, values = [], []
            maxlen = 0
            for indexX in range(min(self.batchSize, len(trainData) - startPosition)):
                for indexY in range(len(trainLabel[indexX + startPosition])):
                    indices.append([indexX, indexY])
                    values.append(int(trainLabel[indexX + startPosition][indexY]))
                if maxlen < len(trainLabel[indexX + startPosition]):
                    maxlen = len(trainLabel[indexX + startPosition])
            shape = [min(self.batchSize, len(trainData) - startPosition), maxlen]

            loss, _ = self.session.run(fetches=[self.parameters['Cost'], self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: (indices, values, shape),
                                                  self.seqLenInput: batachSeq})
            totalLoss += loss

            output = '\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss)
            print(output, end='')
            startPosition += self.batchSize
        return totalLoss
    '''
    def Test(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            result = self.session.run(fetches=self.decodeDense,
                                      feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})

            for indexX in range(len(result)):
                recorder = numpy.zeros(5)
                for indexY in range(len(result[indexX])):
                    recorder[int(result[indexX][indexY])] += 1
                print(recorder)
            exit()
            startPosition += self.batchSize'''
