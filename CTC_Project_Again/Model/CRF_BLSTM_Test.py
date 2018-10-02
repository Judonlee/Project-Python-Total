import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf
import numpy
from __Base.Shuffle import Shuffle


class CRF_BLSTM(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, batchSize=32,
                 learningRate=1e-4, startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.featureShape = featureShape
        self.seqLen = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        super(CRF_BLSTM, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
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
        # RNN Start
        ###################################################################################################

        self.parameters['RNN_Cell_Forward'] = rnn.BasicLSTMCell(num_units=self.hiddenNodules, name='RNN_Cell_Forward')
        self.parameters['RNN_Cell_Backward'] = rnn.BasicLSTMCell(num_units=self.hiddenNodules, name='RNN_Cell_Backward')
        (self.parameters['RNN_Output_Forward'], self.parameters['RNN_Output_Backward']), _ = \
            tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['RNN_Cell_Forward'],
                                                    cell_bw=self.parameters['RNN_Cell_Backward'],
                                                    inputs=self.dataInput, sequence_length=self.seqLenInput,
                                                    dtype=tensorflow.float32)
        self.parameters['RNN_Concat'] = tensorflow.concat(
            (self.parameters['RNN_Output_Forward'], self.parameters['RNN_Output_Backward']), axis=2)

        ###################################################################################################
        # Logits
        ###################################################################################################

        self.parameters['RNN_Reshape'] = tensorflow.reshape(tensor=self.parameters['RNN_Concat'],
                                                            shape=[-1, 2 * self.hiddenNodules], name='RNN_Reshape')
        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Reshape'], units=self.numClass,
                                                            activation=None)
        self.parameters['Logits_Reshape'] = \
            tensorflow.reshape(tensor=self.parameters['Logits'],
                               shape=[self.parameters['BatchSize'], self.parameters['TimeStep'], self.numClass],
                               name='Logits_Reshape')

        ###################################################################################################
        # Conditional Random Field
        ###################################################################################################

        self.parameters['LogLikelihood'], self.parameters['TransitionParams'] = crf.crf_log_likelihood(
            inputs=self.parameters['Logits_Reshape'], tag_indices=self.labelInput, sequence_lengths=self.seqLenInput)

        self.parameters['Loss'] = tensorflow.reduce_mean(input_tensor=-self.parameters['LogLikelihood'], name='Loss')

        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)

        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchData, batchLabel = [], []
            batachSeq = trainSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(trainSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                currentData = numpy.concatenate(
                    (trainData[index], numpy.zeros((maxLen - len(trainData[index]), len(trainData[index][0])))), axis=0)
                batchData.append(currentData)

                currentLabel = numpy.concatenate((trainLabel[index], numpy.zeros(maxLen - len(trainLabel[index]))),
                                                 axis=0)
                batchLabel.append(currentLabel)

            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                  self.seqLenInput: batachSeq})
            totalLoss += loss
            output = '\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss)
            print(output, end='')
            startPosition += self.batchSize
        return totalLoss

    def Test_Decode(self, testData, testLabel, testSeq):
        startPosition = 0
        CorrectLabels = 0
        while startPosition < len(testData):
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
                fetches=[self.parameters['Logits_Reshape'], self.parameters['TransitionParams']],
                feed_dict={self.dataInput: batchData, self.seqLenInput: batchSeq})

            for index in range(len(logits)):
                treatLogits = logits[index][0:batchSeq[index]]
                viterbiSequence, viterbiScore = crf.viterbi_decode(score=treatLogits, transition_params=params)
                # print(viterbiScore, '\nOrigin', testLabel[startPosition + index], '\nResult', viterbiSequence, '\n')

                CorrectLabels += numpy.sum(numpy.equal(viterbiSequence, testLabel[startPosition + index]))

            startPosition += self.batchSize
        print('Correct Rate :', CorrectLabels, numpy.sum(testSeq))
