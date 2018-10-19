import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import tensorflow.contrib.crf as crf
import numpy
from __Base.Shuffle import Shuffle


class CRF_Test(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, batchSize=32,
                 learningRate=1e-4, startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.featureShape = featureShape
        self.seqLen = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        super(CRF_Test, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                       learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                       graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This Model uses the Final_Pooling to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None], name='labelInput')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]
        self.parameters['TimeStep'] = tensorflow.shape(input=self.dataInput, name='TimeStep')[1]

        ###################################################################################################
        # Network Start
        ###################################################################################################

        self.parameters['Data_Reshape'] = tensorflow.reshape(tensor=self.dataInput, shape=[-1, self.featureShape],
                                                             name='Data_Reshape')
        self.parameters['Layer1st_FC'] = tensorflow.layers.dense(inputs=self.parameters['Data_Reshape'],
                                                                 units=self.numClass, activation=tensorflow.nn.tanh,
                                                                 name='Layer1st_FC')
        self.parameters['Logits'] = tensorflow.reshape(tensor=self.parameters['Layer1st_FC'],
                                                       shape=[self.parameters['BatchSize'], self.parameters['TimeStep'],
                                                              self.numClass], name='Logits')

        self.parameters['LogLikelihood'], self.parameters['TransitionParams'] = crf.crf_log_likelihood(
            inputs=self.parameters['Logits'], tag_indices=self.labelInput, sequence_lengths=self.seqLenInput)
        self.parameters['Loss'] = tensorflow.reduce_mean(-self.parameters['LogLikelihood'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)

        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchData, batchLabel = [], []
            batchSeq = trainSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(trainSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                # print(numpy.shape(trainData[index]), numpy.shape(trainLabel[index]))
                currentData = numpy.concatenate(
                    (trainData[index], numpy.zeros((maxLen - len(trainData[index]), len(trainData[index][0])))), axis=0)
                batchData.append(currentData)

                currentLabel = numpy.concatenate((trainLabel[index], numpy.zeros(maxLen - len(trainLabel[index]))),
                                                 axis=0)
                batchLabel.append(currentLabel)

            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                  self.seqLenInput: batchSeq})
            totalLoss += loss
            print('\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss), end='')
            startPosition += self.batchSize
        return totalLoss

    def Test_Decode(self, testData, testLabel, testSeq):
        startPosition = 0
        CorrectLabels = 0
        while startPosition < len(testData):
            print('\rTreating %d/%d' % (startPosition, len(testData)), end='')
            batchData, batchLabel = [], []
            batchSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

                currentLabel = numpy.concatenate((testLabel[index], numpy.zeros(maxLen - len(testLabel[index]))),
                                                 axis=0)
                batchLabel.append(currentLabel)

            [logits, params] = self.session.run(
                fetches=[self.parameters['Logits'], self.parameters['TransitionParams']],
                feed_dict={self.dataInput: batchData, self.seqLenInput: batchSeq})

            for index in range(len(logits)):
                treatLogits = logits[index][0:batchSeq[index]]
                viterbiSequence, viterbiScore = crf.viterbi_decode(score=treatLogits, transition_params=params)
                # print(viterbiScore, '\nOrigin', testLabel[startPosition + index], '\nResult', viterbiSequence, '\n')

                CorrectLabels += numpy.sum(numpy.equal(viterbiSequence, testLabel[startPosition + index]))

            startPosition += self.batchSize
        print('\n\nCorrect Rate :', CorrectLabels, numpy.sum(testSeq))
        return CorrectLabels, numpy.sum(testSeq)

    def Test_Decode_GroundLabel(self, testData, testLabel, testGroundLabel, testSeq):
        startPosition = 0
        CorrectLabels = 0
        matrix = numpy.zeros((self.numClass, self.numClass))
        while startPosition < len(testData):
            print('\rTreating %d/%d' % (startPosition, len(testData)), end='')
            batchData, batchLabel = [], []
            batchSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

                currentLabel = numpy.concatenate((testLabel[index], numpy.zeros(maxLen - len(testLabel[index]))),
                                                 axis=0)
                batchLabel.append(currentLabel)

            [logits, params] = self.session.run(
                fetches=[self.parameters['Logits'], self.parameters['TransitionParams']],
                feed_dict={self.dataInput: batchData, self.seqLenInput: batchSeq})

            for index in range(len(logits)):
                treatLogits = logits[index][0:batchSeq[index]]
                viterbiSequence, viterbiScore = crf.viterbi_decode(score=treatLogits, transition_params=params)

                CorrectLabels += numpy.sum(numpy.equal(viterbiSequence, testLabel[startPosition + index]))

                counter = numpy.zeros(self.numClass)
                for sample in viterbiSequence:
                    counter[sample] += 1
                matrix[numpy.argmax(numpy.array(testGroundLabel[index + startPosition]))][
                    numpy.argmax(numpy.array(counter))] += 1

            startPosition += self.batchSize
        print('\n\nCorrect Rate :', CorrectLabels, numpy.sum(testSeq))
        print(matrix)
        return matrix, CorrectLabels, numpy.sum(testSeq)
