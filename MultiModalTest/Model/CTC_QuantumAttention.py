import tensorflow
from MultiModalTest.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM


class CTC_QuantumAttention(CTC_Multi_BLSTM):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, rnnLayers, hiddenNodules=128,
                 batchSize=32, startFlag=True, graphRevealFlag=False, graphPath='logs/',
                 occupyRate=-1):
        super(CTC_QuantumAttention, self).__init__(
            trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeqLength, featureShape=featureShape,
            numClass=numClass, rnnLayers=rnnLayers, hiddenNodules=hiddenNodules, batchSize=batchSize,
            startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)
        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

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

        (self.parameters['RNN_Output_Forward'], self.parameters['RNN_Output_Backward']), (
            self.parameters['RNN_Forward_FinalState'], self.parameters['RNN_Backward_FinalState']) = \
            tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['Layer_Forward'],
                                                    cell_bw=self.parameters['Layer_Backward'],
                                                    inputs=self.dataInput, sequence_length=self.seqLenInput,
                                                    dtype=tensorflow.float32)
        self.parameters['RNN_Concat'] = tensorflow.concat(
            (self.parameters['RNN_Output_Forward'], self.parameters['RNN_Output_Backward']), axis=2)

        ###################################################################################################
        # Attention
        ###################################################################################################

        with tensorflow.name_scope('QuantumAttentionMechanism'):
            self.parameters['RNN_FinalState'] = tensorflow.concat(
                (self.parameters['RNN_Forward_FinalState'][-1][0], self.parameters['RNN_Backward_FinalState'][-1][0]),
                axis=1, name='RNN_FinalState')
            self.parameters['RNN_StartState'] = self.parameters['RNN_Concat'][:, 0]

            ##############################################################################################

            self.parameters['TotalWeight_Matrix'] = tensorflow.matmul(self.parameters['RNN_FinalState'],
                                                                      self.parameters['RNN_StartState'],
                                                                      transpose_b=True, name='TotalWeight_Matrix')
            self.parameters['TotalWeight'] = tensorflow.matmul(
                tensorflow.multiply(self.parameters['TotalWeight_Matrix'],
                                    tensorflow.eye(num_rows=self.parameters['BatchSize'])),
                tensorflow.ones(shape=[self.parameters['BatchSize'], 1]), name='TotalWeight')

            ##############################################################################################

            self.parameters['RNN_FinalState_Repeat'] = tensorflow.tile(
                input=self.parameters['RNN_FinalState'][:, tensorflow.newaxis, :],
                multiples=[1, self.parameters['TimeStep'], 1], name='RNN_FinalState_Repeat')
            self.parameters['RNN_Weight_UpLeft'] = tensorflow.reduce_sum(
                input_tensor=tensorflow.multiply(self.parameters['RNN_FinalState_Repeat'],
                                                 self.parameters['RNN_Concat']), axis=2, name='RNN_Weight_UpLeft')

            ##############################################################################################

            self.parameters['RNN_StartState_Repeat'] = tensorflow.tile(
                input=self.parameters['RNN_StartState'][:, tensorflow.newaxis, :],
                multiples=[1, self.parameters['TimeStep'], 1], name='RNN_StartState_Repeat')
            self.parameters['RNN_Weight_UpRight'] = tensorflow.reduce_sum(
                input_tensor=tensorflow.multiply(self.parameters['RNN_StartState_Repeat'],
                                                 self.parameters['RNN_Concat']), axis=2, name='RNN_Weight_UpRight')

            ##############################################################################################

            self.parameters['RNN_Weight_Upper'] = tensorflow.multiply(
                self.parameters['RNN_Weight_UpLeft'], self.parameters['RNN_Weight_UpRight'], name='RNN_Weight_Upper')
            self.parameters['RNN_Weight_Downer'] = tensorflow.tile(
                input=self.parameters['TotalWeight'], multiples=[1, self.parameters['TimeStep']],
                name='RNN_Weight_Downer')
            self.parameters['RNN_Weight'] = tensorflow.divide(self.parameters['RNN_Weight_Upper'],
                                                              self.parameters['RNN_Weight_Downer'], name='RNN_Weight')

            self.parameters['Weight_Supplement'] = tensorflow.tile(
                input=tensorflow.reshape(tensor=tensorflow.to_float(self.parameters['TimeStep']), shape=[1, 1]),
                multiples=[self.parameters['BatchSize'], self.parameters['TimeStep']], name='Weight_Supplement')
            self.parameters['RNN_Weight_SoftMax'] = tensorflow.multiply(
                tensorflow.nn.softmax(logits=self.parameters['RNN_Weight'], axis=1),
                self.parameters['Weight_Supplement'], name='RNN_Weight_SoftMax')
            self.parameters['QuantumAttentionWeight'] = tensorflow.tile(
                input=self.parameters['RNN_Weight_SoftMax'][:, :, tensorflow.newaxis],
                multiples=[1, 1, self.hiddenNodules * 2], name='QuantumAttentionWeight')

        self.parameters['RNN_Concat_WithAttention'] = tensorflow.multiply(self.parameters['RNN_Concat'],
                                                                          self.parameters['QuantumAttentionWeight'],
                                                                          name='RNN_Concat_WithAttention')

        self.parameters['RNN_Concat_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['RNN_Concat_WithAttention'],
            shape=[self.parameters['BatchSize'] * self.parameters['TimeStep'], self.hiddenNodules * 2],
            name='RNN_Concat_Reshape')

        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Concat_Reshape'],
                                                            units=self.numClass, activation=None)
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
