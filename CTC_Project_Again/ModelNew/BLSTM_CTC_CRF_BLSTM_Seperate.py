import tensorflow
from CTC_Project_Again.ModelNew.CTC_Multi_BLSTM import CTC_Multi_BLSTM
import numpy
import tensorflow.contrib.crf as crf
from __Base.Shuffle import Shuffle


class BLSTM_CTC_CRF(CTC_Multi_BLSTM):
    def __init__(self, trainData, trainSeqLabel, trainGroundLabel, trainSeqLength, featureShape, numClass, rnnLayers,
                 hiddenNodules=128, batchSize=32, learningRate=1e-3, startFlag=True, graphRevealFlag=False,
                 graphPath='logs/', occupyRate=-1):
        self.groundLabel = trainGroundLabel
        super(BLSTM_CTC_CRF, self).__init__(trainData=trainData, trainLabel=trainSeqLabel,
                                            trainSeqLength=trainSeqLength, featureShape=featureShape, numClass=numClass,
                                            rnnLayers=rnnLayers, hiddenNodules=hiddenNodules, batchSize=batchSize,
                                            learningRate=learningRate, startFlag=startFlag,
                                            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)
        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + '\t' + str(self.parameters[sample])
        # print(self.information)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.sparse_placeholder(dtype=tensorflow.int32, shape=None, name='labelInput')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')
        self.groundLabelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.numClass],
                                                       name='groundLabelInput')

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
        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Reshape'], units=3,
                                                            activation=None)
        self.parameters['Logits_Reshape'] = \
            tensorflow.reshape(tensor=self.parameters['Logits'],
                               shape=[self.parameters['BatchSize'], self.parameters['TimeStep'], 3],
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

        ###################################################################################################
        # CTC Sequence Label
        ###################################################################################################

        self.parameters['CTC_SeqLabel_Voice'] = tensorflow.argmax(
            input=self.parameters['Logits_Reshape'][:, :, 0:2], axis=2, name='CTC_SeqLabel_Voice')
        self.parameters['Ground_Arg'] = tensorflow.add(
            x=tensorflow.argmax(input=self.groundLabelInput, axis=1, name='Ground_Arg'),
            y=tensorflow.ones(shape=[self.parameters['BatchSize']], dtype=tensorflow.int64, name='Arg_Ones'))

        self.parameters['Ground_Arg_Reshape'] = self.parameters['Ground_Arg'][:, tensorflow.newaxis]
        self.parameters['GroundTruthLabel'] = tensorflow.tile(input=self.parameters['Ground_Arg_Reshape'],
                                                              multiples=[1, self.parameters['TimeStep']],
                                                              name='GroundTruthLabel')
        self.parameters['CTC_SeqLabel'] = tensorflow.multiply(x=self.parameters['CTC_SeqLabel_Voice'],
                                                              y=self.parameters['GroundTruthLabel'],
                                                              name='CTC_SeqLabel')

        ###################################################################################################
        # CRF part
        ###################################################################################################

        with tensorflow.variable_scope('CRF_Part_BLSTM'):
            self.parameters['CRF_RNN_Cell_Forward'] = []
            self.parameters['CRF_RNN_Cell_Backward'] = []

            for layers in range(self.rnnLayers):
                self.parameters['CRF_RNN_Cell_Forward'].append(
                    tensorflow.nn.rnn_cell.LSTMCell(num_units=self.hiddenNodules, state_is_tuple=True,
                                                    name='CRF_RNN_Forward_%d' % layers))
                self.parameters['CRF_RNN_Cell_Backward'].append(
                    tensorflow.nn.rnn_cell.LSTMCell(num_units=self.hiddenNodules, state_is_tuple=True,
                                                    name='CRF_RNN_Backward_%d' % layers))

            self.parameters['CRF_Layer_Forward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=self.parameters['CRF_RNN_Cell_Forward'], state_is_tuple=True)
            self.parameters['CRF_Layer_Backward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=self.parameters['CRF_RNN_Cell_Backward'], state_is_tuple=True)

            (self.parameters['CRF_RNN_Output_Forward'], self.parameters['CRF_RNN_Output_Backward']), _ = \
                tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['CRF_Layer_Forward'],
                                                        cell_bw=self.parameters['CRF_Layer_Backward'],
                                                        inputs=self.dataInput, sequence_length=self.seqLenInput,
                                                        dtype=tensorflow.float32)

        self.parameters['CRF_RNN_Concat'] = tensorflow.concat(
            (self.parameters['CRF_RNN_Output_Forward'], self.parameters['CRF_RNN_Output_Backward']), axis=2,
            name='CRF_RNN_Concat')
        self.parameters['CRF_RNN_Reshape'] = tensorflow.reshape(tensor=self.parameters['CRF_RNN_Concat'],
                                                                shape=[-1, 2 * self.hiddenNodules],
                                                                name='CRF_RNN_Reshape')

        self.parameters['CRF_Logits'] = tensorflow.layers.dense(inputs=self.parameters['CRF_RNN_Reshape'],
                                                                units=self.numClass + 1, activation=tensorflow.nn.tanh,
                                                                name='CRF_Logits')
        self.parameters['CRF_Logits_Reshape'] = \
            tensorflow.reshape(tensor=self.parameters['CRF_Logits'],
                               shape=[self.parameters['BatchSize'], self.parameters['TimeStep'], self.numClass + 1],
                               name='CRF_Logits_Reshape')

        ###################################################################################################
        # Conditional Random Field
        ###################################################################################################

        with tensorflow.name_scope('CRF_Loss_Calc'):
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
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.groundLabel, seqLen=self.seqLen)

        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchData = []
            batachSeq = trainSeq[startPosition:startPosition + self.batchSize]
            batchLabel = trainLabel[startPosition:startPosition + self.batchSize]

            maxLen = max(trainSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                currentData = numpy.concatenate(
                    (trainData[index], numpy.zeros((maxLen - len(trainData[index]), len(trainData[index][0])))), axis=0)
                batchData.append(currentData)

            loss, _ = self.session.run(fetches=[self.parameters['CRF_Loss'], self.CRFTrain],
                                       feed_dict={self.dataInput: batchData, self.groundLabelInput: batchLabel,
                                                  self.seqLenInput: batachSeq})
            totalLoss += loss

            output = '\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss)
            print(output, end='')
            startPosition += self.batchSize
        return totalLoss

    def Test_CRF(self, testData, testLabel, testSeq):
        startPosition = 0
        matrix = numpy.zeros((self.numClass, self.numClass))
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

                counter = numpy.zeros(self.numClass + 1)
                for sample in viterbiSequence:
                    counter[sample] += 1
                # print(counter)
                # exit()
                matrix[numpy.argmax(numpy.array(testLabel[index + startPosition]))][
                    numpy.argmax(numpy.array(counter[1:]))] += 1

            startPosition += self.batchSize
        print(matrix)
        return matrix
