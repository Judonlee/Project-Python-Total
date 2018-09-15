import tensorflow
import numpy
import random
from CTC_Project.Module.BaseClass import NeuralNetwork_Base


def Shuffle(data, target, sequence):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newTarget, newSequence = [], [], []
    for sample in index:
        newData.append(data[sample])
        newTarget.append(target[sample])
        newSequence.append(sequence[sample])
    return newData, newTarget, newSequence


class CTC(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, rnnLayers=1,
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
        self.seqLength = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        self.rnnLayer = rnnLayers
        super(CTC, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                  learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                  graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This is the simplest test of CTC Model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='Inputs')
        self.targetsInput = tensorflow.sparse_placeholder(dtype=tensorflow.int32, shape=None, name='Targets')
        self.seqLengthInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=None, name='seqLength')

        self.parameters['Shape'] = tensorflow.shape(self.dataInput, name='Shape')
        self.parameters['BatchSize'], self.parameters['TimeStep'] = self.parameters['Shape'][0], \
                                                                    self.parameters['Shape'][1]

        ###################################################################################################
        # RNN Start
        ###################################################################################################

        self.rnnCell = []
        for layers in range(self.rnnLayer):
            self.parameters['RNN_Cell_Layer' + str(layers)] = tensorflow.contrib.rnn.LSTMCell(self.hiddenNodules)
            self.rnnCell.append(self.parameters['RNN_Cell_Layer' + str(layers)])
        self.parameters['Stack'] = tensorflow.contrib.rnn.MultiRNNCell(self.rnnCell)

        self.parameters['RNN_Outputs'], _ = tensorflow.nn.dynamic_rnn(cell=self.parameters['Stack'],
                                                                      inputs=self.dataInput,
                                                                      sequence_length=self.seqLengthInput,
                                                                      dtype=tensorflow.float32)
        self.parameters['RNN_Outputs_Reshape'] = tensorflow.reshape(tensor=self.parameters['RNN_Outputs'],
                                                                    shape=[-1, self.hiddenNodules],
                                                                    name='RNN_Outputs_Reshape')

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
        self.parameters['Loss'] = tensorflow.nn.ctc_loss(labels=self.targetsInput,
                                                         inputs=self.parameters['Logits_TimeMajor'],
                                                         sequence_length=self.seqLengthInput)
        self.parameters['Cost'] = tensorflow.reduce_mean(self.parameters['Loss'], name='Cost')
        self.train = tensorflow.train.MomentumOptimizer(learning_rate=learningRate, momentum=0.9). \
            minimize(self.parameters['Cost'])

        self.parameters['Decode'], self.parameters['Log_Prob'] = tensorflow.nn.ctc_greedy_decoder(
            inputs=self.parameters['Logits_TimeMajor'], sequence_length=self.seqLengthInput)

    def Train(self):
        trainData, trainTargets, trainSeqLen = Shuffle(data=self.data, target=self.label, sequence=self.seqLength)

        startPosition = 0
        totalLoss = 0.0
        while startPosition < len(trainData):
            batchTrainData, batchIndex, batchValue = [], [], []
            batchTrainSeqLen = trainSeqLen[startPosition:startPosition + self.batchSize]
            maxlen = 0
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                if maxlen < len(trainData[index]): maxlen = len(trainData[index])
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                currentData = numpy.concatenate(
                    (trainData[index], numpy.zeros((maxlen - len(trainData[index]), len(trainData[index][0])))), axis=0)
                batchTrainData.append(currentData)

            maxlen = 0
            for indexX in range(min(self.batchSize, len(trainData) - startPosition)):
                for indexY in range(len(trainTargets[indexX + startPosition])):
                    batchIndex.append([indexX, indexY])
                    batchValue.append(trainTargets[indexX + startPosition][indexY])

                    if maxlen < len(trainTargets[indexX + startPosition]):
                        maxlen = len(trainTargets[indexX + startPosition])
            batchShape = [min(self.batchSize, len(trainData) - startPosition), maxlen]

            loss, _ = self.session.run(fetches=[self.parameters['Cost'], self.train],
                                       feed_dict={self.dataInput: batchTrainData,
                                                  self.targetsInput: (batchIndex, batchValue, batchShape),
                                                  self.seqLengthInput: batchTrainSeqLen})

            string = '\rBatch ' + str(int(startPosition / self.batchSize)) + '/' + str(
                int(len(trainData) / self.batchSize)) + '\tLoss : ' + str(loss)
            print(string, end='')
            totalLoss += loss
            startPosition += self.batchSize
        return totalLoss

    def PredictOutput(self, data, sequence):
        startPosition = 0
        totalLogits = []
        while startPosition < len(data):
            batchData = []
            maxlen = 0
            for index in range(startPosition, min(startPosition + self.batchSize, len(data))):
                if maxlen < len(data[index]): maxlen = len(data[index])
            for index in range(startPosition, min(startPosition + self.batchSize, len(data))):
                currentData = numpy.concatenate(
                    (data[index], numpy.zeros((maxlen - len(data[index]), len(data[index][0])))), axis=0)
                batchData.append(currentData)

            batchSequence = sequence[startPosition:startPosition + self.batchSize]
            currentData, decode = self.session.run(
                fetches=(self.parameters['Logits_Reshape'], self.parameters['Decode'][0]),
                feed_dict={self.dataInput: batchData, self.seqLengthInput: batchSequence})
            print(numpy.shape(currentData))

            file = open('Logits.csv', 'w')
            for indexX in range(len(currentData)):
                for indexY in range(len(currentData[indexX])):
                    file.write(str(numpy.argmax(numpy.array(currentData[indexX][indexY]))) + ',')
                file.write('\n')
            file.close()

            print(decode)
            data = numpy.zeros(decode[2])
            for index in range(len(decode[0])):
                data[decode[0][index][0]][decode[0][index][1]] = decode[1][index]
            file = open('Decode.csv', 'w')
            for indexX in range(len(data)):
                for indexY in range(len(data[indexX])):
                    file.write(str(data[indexX][indexY]) + ',')
                file.write('\n')
            file.close()
            exit()
            startPosition += self.batchSize
