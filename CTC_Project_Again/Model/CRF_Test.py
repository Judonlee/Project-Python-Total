import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import tensorflow.contrib.rnn as rnn
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

        self.parameters['ViterbiSequence'], self.parameters['ViterbiScore'] = crf.crf_decode(
            potentials=self.parameters['Logits_Reshape'], transition_params=self.parameters['TransitionParams'],
            sequence_length=self.seqLenInput)
        self.parameters['Loss'] = tensorflow.reduce_mean(input_tensor=self.parameters['LogLikelihood'], name='Loss')

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

    def Test_CRF(self, testData, testLabel, testSeq):
        startPosition = 0
        totalPredict = []
        totalScore = []
        while startPosition < len(testData):
            batchData, batchLabel = [], []
            batachSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                output = '\rBatch : %d/%d' % (startPosition, len(testData))
                print(output, end='')

                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

                currentLabel = numpy.concatenate((testLabel[index], numpy.zeros(maxLen - len(testLabel[index]))),
                                                 axis=0)
                batchLabel.append(currentLabel)

            result, score = self.session.run(
                fetches=[self.parameters['ViterbiSequence'], self.parameters['ViterbiScore']],
                feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})
            for index in range(len(batachSeq)):
                totalPredict.append(result[index][0:batachSeq[index]])
                totalScore.append(score[index])

            startPosition += self.batchSize

        print('\n\n')
        matrix = numpy.zeros((self.numClass, self.numClass))
        for index in range(len(totalPredict)):
            notebook = numpy.zeros(5)
            for sample in totalPredict[index]:
                notebook[sample] += 1
            # print(notebook, numpy.argmax(numpy.array(notebook)), totalScore[index], testLabel[index][0])

            matrix[int(testLabel[index][0])][numpy.argmax(numpy.array(notebook)[1:5])+1] += 1
        print(matrix)
