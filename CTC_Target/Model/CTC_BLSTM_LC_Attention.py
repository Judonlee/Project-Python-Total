import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import numpy
from CTC_Target.Loader.IEMOCAP_Loader import Load
from __Base.Shuffle import Shuffle


class CTC_LC_Attention(CTC_Multi_BLSTM):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, rnnLayers, attentionScope,
                 hiddenNodules=128, batchSize=32, learningRate=1e-3, startFlag=True, graphRevealFlag=False,
                 graphPath='logs/', occupyRate=-1):
        self.attentionScope = attentionScope
        super(CTC_LC_Attention, self).__init__(trainData=trainData, trainLabel=trainLabel,
                                               trainSeqLength=trainSeqLength, featureShape=featureShape,
                                               numClass=numClass, rnnLayers=rnnLayers, hiddenNodules=hiddenNodules,
                                               batchSize=batchSize, learningRate=learningRate, startFlag=startFlag,
                                               graphRevealFlag=graphRevealFlag, graphPath=graphPath,
                                               occupyRate=occupyRate)
        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + '\t' + str(self.parameters[sample])

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

        self.parameters['RNN_Cell_Forward'] = []
        self.parameters['RNN_Cell_Backward'] = []

        for layers in range(self.rnnLayers):
            self.parameters['RNN_Cell_Forward'].append(
                tensorflow.nn.rnn_cell.LSTMCell(num_units=self.hiddenNodules, state_is_tuple=True,
                                                name='RNN_Cell_Forward_%d' % layers))
            self.parameters['RNN_Cell_Backward'].append(
                tensorflow.nn.rnn_cell.LSTMCell(num_units=self.hiddenNodules, state_is_tuple=True,
                                                name='RNN_Cell_Backward_%d' % layers))

        self.parameters['Layer_Forward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=self.parameters['RNN_Cell_Forward'], state_is_tuple=True)
        self.parameters['Layer_Backward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=self.parameters['RNN_Cell_Backward'], state_is_tuple=True)

        (self.parameters['RNN_Output_Forward'], self.parameters['RNN_Output_Backward']), _ = \
            tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['Layer_Forward'],
                                                    cell_bw=self.parameters['Layer_Backward'],
                                                    inputs=self.dataInput, sequence_length=self.seqLenInput,
                                                    dtype=tensorflow.float32)
        self.parameters['RNN_Concat'] = tensorflow.concat(
            (self.parameters['RNN_Output_Forward'], self.parameters['RNN_Output_Backward']), axis=2)

        ###################################################################################################
        # Logits & Attention
        ###################################################################################################

        self.parameters['RNN_Reshape'] = tensorflow.reshape(tensor=self.parameters['RNN_Concat'],
                                                            shape=[-1, 2 * self.hiddenNodules], name='RNN_Reshape')

        self.parameters['Attention_Value'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Reshape'], units=1,
                                                                     activation=tensorflow.nn.tanh,
                                                                     name='Attention_Value')
        with tensorflow.name_scope('Attention_Concat'):
            self.parameters['Attention_Concat'] = tensorflow.concat(values=[
                self.parameters['Attention_Value'][0:-self.attentionScope],
                self.parameters['Attention_Value'][1:-self.attentionScope + 1]], axis=1)

            for concatCounter in range(2, self.attentionScope):
                self.parameters['Attention_Concat'] = tensorflow.concat(
                    values=[self.parameters['Attention_Concat'],
                            self.parameters['Attention_Value'][concatCounter:-self.attentionScope + concatCounter]],
                    axis=1)

        self.parameters['Attention_Evaluate'] = tensorflow.nn.softmax(logits=self.parameters['Attention_Concat'],
                                                                      name='Attention_Evaluate')

        with tensorflow.name_scope('AttentionAdd'):
            self.parameters['MultiPart_%04d' % 0] = tensorflow.tile(input=self.parameters['Attention_Evaluate'][:, 0:1],
                                                                    multiples=[1, 2 * self.hiddenNodules],
                                                                    name='MultiPart_%04d' % 0)
            self.parameters['RNN_WithAttention_Total'] = tensorflow.multiply(
                x=self.parameters['RNN_Reshape'][0:-self.attentionScope], y=self.parameters['MultiPart_%04d' % 0])

            for MultiCounter in range(1, self.attentionScope):
                self.parameters['MultiPart_%04d' % MultiCounter] = tensorflow.tile(
                    input=self.parameters['Attention_Evaluate'][:, MultiCounter:MultiCounter + 1],
                    multiples=[1, 2 * self.hiddenNodules],
                    name='MultiPart_%04d' % 0)
                self.parameters['RNN_WithAttention_%04d' % MultiCounter] = tensorflow.multiply(
                    x=self.parameters['RNN_Reshape'][MultiCounter:-self.attentionScope + MultiCounter],
                    y=self.parameters['MultiPart_%04d' % MultiCounter])
                self.parameters['RNN_WithAttention_Total'] = tensorflow.add(
                    x=self.parameters['RNN_WithAttention_Total'],
                    y=self.parameters['RNN_WithAttention_%04d' % MultiCounter])

        self.parameters['RNN_Final'] = tensorflow.concat(
            values=[self.parameters['RNN_WithAttention_Total'],
                    tensorflow.zeros(shape=[self.attentionScope, 2 * self.hiddenNodules])], axis=0)

        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Final'], units=self.numClass,
                                                            activation=None)
        self.parameters['Logits_Reshape'] = \
            tensorflow.reshape(tensor=self.parameters['Logits'],
                               shape=[self.parameters['BatchSize'], self.parameters['TimeStep'], self.numClass],
                               name='Logits_Reshape')

        self.parameters['Logits_TimeMajor'] = tensorflow.transpose(a=self.parameters['Logits_Reshape'], perm=(1, 0, 2),
                                                                   name='Logits_TimeMajor')

        ###################################################################################################
        # CTC part
        ###################################################################################################

        self.parameters['Loss'] = tensorflow.nn.ctc_loss(labels=self.labelInput,
                                                         inputs=self.parameters['Logits_TimeMajor'],
                                                         sequence_length=self.seqLenInput,
                                                         ignore_longer_outputs_than_inputs=True)
        self.parameters['Cost'] = tensorflow.reduce_mean(self.parameters['Loss'], name='Cost')
        self.train = tensorflow.train.RMSPropOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Cost'])
        self.decode, self.logProbability = tensorflow.nn.ctc_beam_search_decoder(
            inputs=self.parameters['Logits_TimeMajor'], sequence_length=self.seqLenInput, merge_repeated=False)
        self.decodeDense = tensorflow.sparse_tensor_to_dense(sp_input=self.decode[0])

    def Train(self):
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)

        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchData = []
            batachSeq = trainSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(trainSeq[startPosition:startPosition + self.batchSize]) + self.attentionScope
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

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize]) + self.attentionScope
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
                records = numpy.zeros(self.numClass - 1)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][0:self.numClass - 1]
                    records[numpy.argmax(numpy.array(chooseArera))] += 1
                totalPredictLogits.append(records)

            ####################################################################
            # 第三部分
            ####################################################################

            for indexX in range(numpy.shape(logits)[0]):
                records = numpy.zeros(self.numClass - 1)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][0:self.numClass - 1]
                    totalSoftMax = numpy.sum(numpy.exp(chooseArera))
                    for indexZ in range(self.numClass - 1):
                        records[indexZ] += numpy.exp(chooseArera[indexZ]) / totalSoftMax

                for indexZ in range(self.numClass - 1):
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
        # print(matrixDecode)
        # print(matrixLogits)
        # print(matrixSoftMax)
        return matrixDecode, matrixLogits, matrixSoftMax

    def Test_AllMethodsWithLen(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredictDecode, totalPredictLogits, totalPredictSoftMax = [], [], []
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize]) + self.attentionScope
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
                records = numpy.zeros(self.numClass - 1)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][0:self.numClass - 1]
                    records[numpy.argmax(numpy.array(chooseArera))] += 1
                totalPredictLogits.append(records)

            ####################################################################
            # 第三部分
            ####################################################################

            for indexX in range(numpy.shape(logits)[0]):
                records = numpy.zeros(self.numClass - 1)
                for indexY in range(testSeq[startPosition + indexX]):
                    chooseArera = logits[indexX][indexY][0:self.numClass - 1]
                    totalSoftMax = numpy.sum(numpy.exp(chooseArera))
                    for indexZ in range(self.numClass - 1):
                        records[indexZ] += numpy.exp(chooseArera[indexZ]) / totalSoftMax

                for indexZ in range(self.numClass - 1):
                    records[indexZ] /= testSeq[startPosition + indexX]
                totalPredictSoftMax.append(records)

            startPosition += self.batchSize

        matrixDecode, matrixLogits, matrixSoftMax = numpy.zeros((self.numClass - 1, self.numClass - 1)), \
                                                    numpy.zeros((self.numClass - 1, self.numClass - 1)), \
                                                    numpy.zeros((self.numClass - 1, self.numClass - 1))

        for index in range(len(totalPredictDecode)):
            matrixDecode[numpy.argmax(numpy.array(testLabel[index]))][totalPredictDecode[index]] += testSeq[index]
        for index in range(len(totalPredictLogits)):
            matrixLogits[numpy.argmax(numpy.array(testLabel[index]))][
                numpy.argmax(numpy.array(totalPredictLogits[index]))] += testSeq[index]
        for index in range(len(totalPredictSoftMax)):
            matrixSoftMax[numpy.argmax(numpy.array(testLabel[index]))][
                numpy.argmax(numpy.array(totalPredictSoftMax[index]))] += testSeq[index]
        # print(matrixDecode)
        # print(matrixLogits)
        # print(matrixSoftMax)
        return matrixDecode, matrixLogits, matrixSoftMax


if __name__ == '__main__':
    bands = 30
    loadpath = 'D:/ProjectData/CTC_Target/Features/Bands%d/' % bands
    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load(
        loadpath=loadpath, appoint=1)

    classifier = CTC_LC_Attention(trainData=testData, trainLabel=testlabel, trainSeqLength=testSeq, featureShape=30,
                                  numClass=5, rnnLayers=2, attentionScope=5, graphRevealFlag=True)
    print(classifier.information)
    classifier.Train()
