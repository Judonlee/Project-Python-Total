import tensorflow
from MultiModalTest.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
import numpy


class CTC_COMA_Attention(CTC_LC_Attention):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, rnnLayers, attentionScope,
                 hiddenNodules=128, batchSize=32, startFlag=True, graphRevealFlag=False, graphPath='logs/',
                 occupyRate=-1):
        super(CTC_COMA_Attention, self).__init__(
            trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeqLength, featureShape=featureShape,
            numClass=numClass, rnnLayers=rnnLayers, attentionScope=attentionScope, hiddenNodules=hiddenNodules,
            batchSize=batchSize, startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath,
            occupyRate=occupyRate)
        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + '\t' + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.sparse_placeholder(dtype=tensorflow.int32, shape=None, name='labelInput')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')
        self.learningRate = tensorflow.placeholder(dtype=tensorflow.float32, name='learningRate')

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

        self.parameters['RNN_Reshape'] = tensorflow.reshape(tensor=self.parameters['RNN_Concat'], shape=[
            self.parameters['BatchSize'] * self.parameters['TimeStep'], 2 * self.hiddenNodules], name='RNN_Reshape')
        self.parameters['RNN_Transpose'] = tensorflow.transpose(self.parameters['RNN_Reshape'], perm=[1, 0],
                                                                name='RNN_Transpose')[:, :, tensorflow.newaxis]

        with tensorflow.name_scope('AttentionPart'):
            self.parameters['SeqLink'] = tensorflow.concat(
                values=[self.parameters['RNN_Transpose'][:, 0:-self.attentionScope, :],
                        self.parameters['RNN_Transpose'][:, 1:-self.attentionScope + 1, :]], axis=2)
            for concatCounter in range(2, self.attentionScope):
                self.parameters['SeqLink'] = tensorflow.concat(
                    values=[self.parameters['SeqLink'],
                            self.parameters['RNN_Transpose'][:, concatCounter:-self.attentionScope + concatCounter, :]],
                    axis=2)
            self.parameters['AttentionValue_Before'] = tensorflow.nn.softmax(logits=self.parameters['SeqLink'], axis=2,
                                                                             name='AttentionValue_Before')
            self.parameters['AttentionValue'] = tensorflow.transpose(self.parameters['AttentionValue_Before'],
                                                                     perm=[1, 0, 2], name='AttentionValue')

        ###################################################################################################

        with tensorflow.name_scope('WithWeights'):
            self.parameters['RNN_AfterAttention'] = tensorflow.multiply(
                x=self.parameters['RNN_Reshape'][0:-self.attentionScope],
                y=self.parameters['AttentionValue'][:, :, 0])
            for MultiCounter in range(1, self.attentionScope):
                self.parameters['RNN_AfterAttention_Step%d' % MultiCounter] = tensorflow.multiply(
                    x=self.parameters['RNN_Reshape'][MultiCounter:-self.attentionScope + MultiCounter],
                    y=self.parameters['AttentionValue'][:, :, MultiCounter])
                self.parameters['RNN_AfterAttention'] = tensorflow.add(
                    x=self.parameters['RNN_AfterAttention'],
                    y=self.parameters['RNN_AfterAttention_Step%d' % MultiCounter])

        self.parameters['RNN_Final'] = tensorflow.concat(
            values=[self.parameters['RNN_AfterAttention'],
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
        self.train = tensorflow.train.RMSPropOptimizer(learning_rate=self.learningRate).minimize(
            self.parameters['Cost'])
        self.decode, self.logProbability = tensorflow.nn.ctc_beam_search_decoder(
            inputs=self.parameters['Logits_TimeMajor'], sequence_length=self.seqLenInput, merge_repeated=False)
        self.decodeDense = tensorflow.sparse_tensor_to_dense(sp_input=self.decode[0])
