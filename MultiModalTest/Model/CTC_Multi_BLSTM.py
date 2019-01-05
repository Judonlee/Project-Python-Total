import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy
from __Base.Shuffle import Shuffle


class CTC_Multi_BLSTM(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, rnnLayers, hiddenNodules=128,
                 batchSize=32, learningRate=1e-3, startFlag=True, graphRevealFlag=False, graphPath='logs/',
                 occupyRate=-1):
        '''
        :param trainLabel:      In this case, trainLabel is the targets.
        :param trainSeqLength:  In this case, the trainSeqLength are needed which is the length of each cases.
        :param featureShape:    designate how many features in one vector.
        :param hiddenNodules:   designite the number of hidden nodules.
        :param numClass:        designite the number of classes
        '''

        self.featureShape = featureShape
        self.seqLen = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        self.rnnLayers = rnnLayers
        super(CTC_Multi_BLSTM, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                              learningRate=learningRate, startFlag=startFlag,
                                              graphRevealFlag=graphRevealFlag,
                                              graphPath=graphPath, occupyRate=occupyRate)
        self.information = ''
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
