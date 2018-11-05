import tensorflow
from CTC_Project_Again.ModelNew.CTC_Single_Origin import CTC_BLSTM
import numpy
from __Base.Shuffle import Shuffle
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf


class BLSTM_CTC_CRF(CTC_BLSTM):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, rnnLayers, hiddenNodules=128,
                 batchSize=64, learningRate=1e-3, startFlag=True, graphRevealFlag=False, graphPath='logs/',
                 occupyRate=-1):
        '''
        :param trainLabel:      In this case, trainLabel is the targets.
        :param trainSeqLength:  In this case, the trainSeqLength are needed which is the length of each cases.
        :param featureShape:    designate how many features in one vector.
        :param hiddenNodules:   designite the number of hidden nodules.
        :param numClass:        designite the number of classes
        '''
        self.rnnLayers = rnnLayers
        super(BLSTM_CTC_CRF, self).__init__(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeqLength,
                                            featureShape=featureShape, numClass=numClass, hiddenNodules=hiddenNodules,
                                            batchSize=batchSize, learningRate=learningRate, startFlag=startFlag,
                                            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)
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

        self.CRFTrain = tensorflow.train.AdamOptimizer(learning_rate=learningRate). \
            minimize(self.parameters['CRF_Loss'], var_list=tensorflow.global_variables()[6 + 12 * self.rnnLayers:])

    def Load_CTC(self, loadpath):
        saver = tensorflow.train.Saver(var_list=tensorflow.global_variables()[0:6 + 12 * self.rnnLayers])
        saver.restore(self.session, loadpath)

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

            loss, _ = self.session.run(fetches=[self.parameters['CRF_Loss'], self.CRFTrain],
                                       feed_dict={self.dataInput: batchData, self.labelInput: (indices, values, shape),
                                                  self.seqLenInput: batachSeq})
            totalLoss += loss

            output = '\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss)
            print(output, end='')
            startPosition += self.batchSize
        return totalLoss

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
                fetches=[self.parameters['CRF_Logits_Reshape'], self.parameters['CRF_TransitionParams']],
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
