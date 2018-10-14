import tensorflow
from CTC_Project_Again.Model.CTC import CTC
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf
import numpy
from __Base.Shuffle import Shuffle


class CTC_CRF_Reuse(CTC):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, rnnLayers=1,
                 batchSize=64, learningRate=1e-3, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        super(CTC_CRF_Reuse, self).__init__(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeqLength,
                                            featureShape=featureShape, batchSize=batchSize, numClass=numClass,
                                            hiddenNodules=hiddenNodules, rnnLayers=rnnLayers, learningRate=learningRate,
                                            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath,
                                            occupyRate=occupyRate)

        self.information = 'This Model uses the CTC_CRF_Reuse to testify the validation of the model.'
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

        self.parameters['Logits_TimeMajor'] = tensorflow.transpose(a=self.parameters['Logits_Reshape'], perm=(1, 0, 2),
                                                                   name='Logits_TimeMajor')

        ###################################################################################################
        # CTC part
        ###################################################################################################

        self.parameters['Loss'] = tensorflow.nn.ctc_loss(labels=self.labelInput,
                                                         inputs=self.parameters['Logits_TimeMajor'],
                                                         sequence_length=self.seqLenInput)
        self.parameters['Cost'] = tensorflow.reduce_mean(self.parameters['Loss'], name='Cost')
        self.train = tensorflow.train.RMSPropOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Cost'])
        self.decode, self.logProbability = tensorflow.nn.ctc_beam_search_decoder(
            inputs=self.parameters['Logits_TimeMajor'], sequence_length=self.seqLenInput, merge_repeated=False)
        self.decodeDense = tensorflow.sparse_tensor_to_dense(sp_input=self.decode[0])

        ###################################################################################################
        # CTC Sequence Label
        ###################################################################################################

        self.parameters['CTC_SeqLabel'] = tensorflow.argmax(
            input=self.parameters['Logits_Reshape'][:, :, 0:self.numClass - 1], axis=2, name='CTC_SeqLabel')

        ###################################################################################################
        # CRF part
        ###################################################################################################

        self.parameters['CRF_Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Reshape'],
                                                                units=self.numClass - 1, activation=tensorflow.nn.tanh,
                                                                name='CRF_Logits')
        self.parameters['CRF_Logits_Reshape'] = \
            tensorflow.reshape(tensor=self.parameters['CRF_Logits'],
                               shape=[self.parameters['BatchSize'], self.parameters['TimeStep'], self.numClass - 1],
                               name='CRF_Logits_Reshape')

        ###################################################################################################
        # Conditional Random Field
        ###################################################################################################

        self.parameters['CRF_LogLikelihood'], self.parameters['CRF_TransitionParams'] = crf.crf_log_likelihood(
            inputs=self.parameters['CRF_Logits_Reshape'], tag_indices=self.parameters['CTC_SeqLabel'],
            sequence_lengths=self.seqLenInput)

        self.parameters['CRF_Loss'] = tensorflow.reduce_mean(input_tensor=-self.parameters['CRF_LogLikelihood'],
                                                             name='CRF_Loss')
        self.parameters['TotalLoss'] = tensorflow.add(x=self.parameters['Cost'], y=self.parameters['CRF_Loss'],
                                                      name='TotalLoss')

        self.CRFTrain = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            self.parameters['TotalLoss'])

    def CRF_Train(self):
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

            loss, _ = self.session.run(fetches=[self.parameters['TotalLoss'], self.CRFTrain],
                                       feed_dict={self.dataInput: batchData, self.labelInput: (indices, values, shape),
                                                  self.seqLenInput: batachSeq})
            totalLoss += loss

            output = '\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss)
            print(output, end='')
            startPosition += self.batchSize
        return totalLoss

    def LoadPart(self, loadpath):
        saver = tensorflow.train.Saver(var_list=tensorflow.global_variables()[0:18])
        saver.restore(self.session, loadpath)

    def Test_CRF(self, testData, testLabel, testSeq):
        startPosition = 0
        matrix = numpy.zeros((self.numClass - 1, self.numClass - 1))
        while startPosition < len(testData):
            print('\rTesting %d/%d' % (startPosition, len(testSeq)), end='')
            batchData = []
            batchSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            [logits, params] = self.session.run(
                fetches=[self.parameters['Loss'], self.parameters['CRF_TransitionParams']],
                feed_dict={self.dataInput: batchData, self.seqLenInput: batchSeq})

            for index in range(len(logits)):
                treatLogits = logits[index][0:batchSeq[index]]
                viterbiSequence, viterbiScore = crf.viterbi_decode(score=treatLogits, transition_params=params)

                counter = numpy.zeros(self.numClass)
                for sample in viterbiSequence:
                    counter[sample] += 1
                matrix[numpy.argmax(numpy.array(testLabel[index + startPosition]))][
                    numpy.argmax(numpy.array(counter))] += 1

            startPosition += self.batchSize
        print(matrix)
        return matrix
