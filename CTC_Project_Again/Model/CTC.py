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
        self.decodeDense = tensorflow.sparse_tensor_to_dense(sp_input=self.decode[0], default_value=4)

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

    def Test_AllMethods(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredictDecode, totalPredictLogits, totalPredictSoftMax = [], [], []
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            decode, logits = self.session.run(fetches=[self.decode, self.parameters['Logits_Reshape']],
                                              feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})
            ####################################################################
            # 第一部分
            ####################################################################

            indices, value = decode[0].indices, decode[0].values
            result = numpy.zeros((len(batchData), self.numClass))
            for index in range(len(value)):
                result[indices[index][0]][value[index]] += 1
            for sample in result:
                totalPredictDecode.append(numpy.argmax(numpy.array(sample)))

            ####################################################################
            # 第二部分
            ####################################################################

            for indexX in range(numpy.shape(logits)[0]):
                records = numpy.zeros(4)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][0:self.numClass - 1]
                    records[numpy.argmax(numpy.array(chooseArera))] += 1
                totalPredictLogits.append(records)

            ####################################################################
            # 第三部分
            ####################################################################

            for indexX in range(numpy.shape(logits)[0]):
                records = numpy.zeros(4)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][0:self.numClass - 1]
                    totalSoftMax = numpy.sum(numpy.exp(chooseArera))
                    for indexZ in range(4):
                        records[indexZ] += numpy.exp(chooseArera[indexZ]) / totalSoftMax

                for indexZ in range(4):
                    records[indexZ] /= testSeq[startPosition + indexX]
                totalPredictSoftMax.append(records)

            startPosition += self.batchSize

        matrixDecode, matrixLogits, matrixSoftMax = numpy.zeros((self.numClass - 1, self.numClass - 1)), \
                                                    numpy.zeros((self.numClass - 1, self.numClass - 1)), \
                                                    numpy.zeros((self.numClass - 1, self.numClass - 1))

        for index in range(len(totalPredictDecode)):
            matrixDecode[numpy.argmax(numpy.array(testLabel[index]))][totalPredictDecode[index]] += 1
        for index in range(len(totalPredictLogits)):
            matrixLogits[numpy.argmax(numpy.array(testLabel[index]))][
                numpy.argmax(numpy.array(totalPredictLogits[index]))] += 1
        for index in range(len(totalPredictSoftMax)):
            matrixSoftMax[numpy.argmax(numpy.array(testLabel[index]))][
                numpy.argmax(numpy.array(totalPredictSoftMax[index]))] += 1
        return matrixDecode, matrixLogits, matrixSoftMax

    def Test_Decode(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)
            # print(numpy.shape(batchData))
            decode = self.session.run(fetches=self.decode,
                                      feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})
            indices, value = decode[0].indices, decode[0].values

            result = numpy.zeros((len(batchData), self.numClass))
            for index in range(len(value)):
                result[indices[index][0]][value[index]] += 1
            for sample in result:
                totalPredict.append(numpy.argmax(numpy.array(sample)))

            startPosition += self.batchSize

        matrix = numpy.zeros((4, 4))
        for index in range(len(totalPredict)):
            # print(testLabel[index], totalPredict[index])
            matrix[numpy.argmax(numpy.array(testLabel[index]))][totalPredict[index]] += 1
        print()
        print(matrix)
        return matrix

    def Test_LogitsPooling(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            logits = self.session.run(fetches=self.parameters['Logits_Reshape'],
                                      feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})

            for indexX in range(numpy.shape(logits)[0]):
                records = numpy.zeros(4)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][0:4]
                    records[numpy.argmax(numpy.array(chooseArera))] += 1
                # print(records)
                totalPredict.append(records)
            startPosition += self.batchSize
            # exit()
        print()

        matrix = numpy.zeros((4, 4))
        for index in range(len(testLabel)):
            # print(testLabel[index], totalPredict[index])
            matrix[numpy.argmax(numpy.array(testLabel[index]))][numpy.argmax(numpy.array(totalPredict[index]))] += 1
        print(matrix)
        return matrix

    def Test_SoftMax(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            logits = self.session.run(fetches=self.parameters['Logits_Reshape'],
                                      feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})

            for indexX in range(numpy.shape(logits)[0]):
                records = numpy.zeros(4)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][0:4]

                    totalSoftMax = numpy.sum(numpy.exp(chooseArera))
                    for indexZ in range(4):
                        records[indexZ] += numpy.exp(chooseArera[indexZ]) / totalSoftMax

                for indexZ in range(4):
                    records[indexZ] /= testSeq[startPosition + indexX]
                totalPredict.append(records)
                # print(records)
            startPosition += self.batchSize
        print()
        # exit()
        matrix = numpy.zeros((4, 4))
        for index in range(len(testLabel)):
            # print(testLabel[index], totalPredict[index])
            matrix[numpy.argmax(numpy.array(testLabel[index]))][numpy.argmax(numpy.array(totalPredict[index]))] += 1
        print(matrix)
        return matrix

    def LogitsOutput(self, testData, testSeq):
        startPosition = 0
        totalResult = []
        while startPosition < len(testData):
            print('\rCalculating %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            logits = self.session.run(fetches=self.parameters['Logits_Reshape'],
                                      feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})
            for index in range(min(self.batchSize, len(testData) - startPosition)):
                totalResult.append(logits[index][0:testSeq[startPosition + index]])

            startPosition += self.batchSize
        return totalResult

    def Test_Decode_Class6(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)
            # print(numpy.shape(batchData))
            decode = self.session.run(fetches=self.decode,
                                      feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})
            indices, value = decode[0].indices, decode[0].values

            batchPredict = numpy.zeros((self.batchSize, self.numClass))
            for searchIndex in range(len(indices)):
                batchPredict[indices[searchIndex][0]][value[searchIndex]] += 1

            for index in range(min(self.batchSize, len(testData) - startPosition)):
                totalPredict.append(numpy.argmax(numpy.array(batchPredict[index][1:5])))
            startPosition += self.batchSize

        print()
        totalMatrix = numpy.zeros((4, 4))
        for index in range(len(testLabel)):
            totalMatrix[numpy.argmax(numpy.array(testLabel[index]))][totalPredict[index]] += 1
        print(totalMatrix)
        return totalMatrix

    def Test_LogitsPooling_Class6(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            logits = self.session.run(fetches=self.parameters['Logits_Reshape'],
                                      feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})

            for indexX in range(numpy.shape(logits)[0]):
                records = numpy.zeros(4)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][1:5]
                    records[numpy.argmax(numpy.array(chooseArera))] += 1
                totalPredict.append(records)
            startPosition += self.batchSize

        print()
        totalMatrix = numpy.zeros((4, 4))
        for index in range(len(testLabel)):
            totalMatrix[numpy.argmax(numpy.array(testLabel[index]))][
                numpy.argmax(numpy.array(totalPredict[index]))] += 1
        print(totalMatrix)
        return totalMatrix

    def Test_SoftMax_Class6(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            logits = self.session.run(fetches=self.parameters['Logits_Reshape'],
                                      feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})

            for indexX in range(numpy.shape(logits)[0]):
                records = numpy.zeros(4)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][1:5]

                    totalSoftMax = numpy.sum(numpy.exp(chooseArera))
                    for indexZ in range(4):
                        records[indexZ] += numpy.exp(chooseArera[indexZ]) / totalSoftMax

                for indexZ in range(4):
                    records[indexZ] /= testSeq[startPosition + indexX]
                totalPredict.append(records)
                # print(records)
            startPosition += self.batchSize
        print()
        # exit()
        matrix = numpy.zeros((4, 4))
        for index in range(len(testLabel)):
            # print(testLabel[index], totalPredict[index])
            matrix[numpy.argmax(numpy.array(testLabel[index]))][numpy.argmax(numpy.array(totalPredict[index]))] += 1
        print(matrix)
        return matrix
