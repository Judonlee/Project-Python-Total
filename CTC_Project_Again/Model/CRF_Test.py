import tensorflow
from CTC_Project_Again.Model.CTC import CTC
import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.crf as crf


class CRF_Test(CTC):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, batchSize=32,
                 learningRate=1e-4, startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        super(CRF_Test, self).__init__(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeqLength,
                                       featureShape=featureShape, batchSize=batchSize, numClass=numClass,
                                       hiddenNodules=hiddenNodules, learningRate=learningRate, startFlag=startFlag,
                                       graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This model is to testify the ability of CRF Model.'
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

        self.parameters['LogLikelihood'], self.parameters['TransitionParams'] = crf.crf_log_likelihood(
            inputs=self.parameters['Logits_Reshape'], tag_indices=self.labelInput, sequence_lengths=self.seqLenInput)

        self.parameters['ViterbiSequence'], self.parameters['ViterbiScore'] = crf.crf_decode(
            potentials=self.parameters['Logits_Reshape'], transition_params=self.parameters['TransitionParams'],
            sequence_length=self.seqLenInput)
        self.parameters['Loss'] = tensorflow.reduce_mean(input_tensor=self.parameters['LogLikelihood'], name='Loss')
