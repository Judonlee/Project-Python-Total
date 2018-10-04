import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy
from tensorflow.contrib import rnn
from tensorflow.contrib import crf
from __Base.Shuffle import Shuffle


class BLSTM_CTC_BLSTM_CRF(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, batchSize=32, learningRate=1e-4,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        '''
        :param trainLabel:      In this case, trainLabel is the targets.
        :param trainSeqLength:  In this case, the trainSeqLength are needed which is the length of each cases.
        :param featureShape:    designate how many features in one vector.
        :param numClass:        designite the number of classes
        '''

        self.featureShape = featureShape
        self.seqLen = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = 128
        super(BLSTM_CTC_BLSTM_CRF, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                                  learningRate=learningRate, startFlag=startFlag,
                                                  graphRevealFlag=graphRevealFlag, graphPath=graphPath,
                                                  occupyRate=occupyRate)

        self.information = 'This Model uses the BLSTM_CTC_BLSTM_CRF to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.sparse_placeholder(dtype=tensorflow.int32, shape=None, name='labelInputCTC')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]
        self.parameters['TimeStep'] = tensorflow.shape(input=self.dataInput, name='TimeStep')[1]

        ###################################################################################################
        # CTC BLSTM Part
        ###################################################################################################

        with tensorflow.variable_scope('CTC_BLSTM'):
            self.parameters['CTC_Cell_Forward'] = rnn.BasicLSTMCell(num_units=self.hiddenNodules,
                                                                    name='CTC_Cell_Forward')
            self.parameters['CTC_Cell_Backward'] = rnn.BasicLSTMCell(num_units=self.hiddenNodules,
                                                                     name='CTC_Cell_Backward')
            (self.parameters['CTC_Output_Forward'], self.parameters['CTC_Output_Backward']), _ = \
                tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['CTC_Cell_Forward'],
                                                        cell_bw=self.parameters['CTC_Cell_Backward'],
                                                        inputs=self.dataInput, sequence_length=self.seqLenInput,
                                                        dtype=tensorflow.float32)
            self.parameters['CTC_Concat'] = tensorflow.concat(
                (self.parameters['CTC_Output_Forward'], self.parameters['CTC_Output_Backward']), axis=2)

        ###################################################################################################
        # CTC Logits Part
        ###################################################################################################

        with tensorflow.name_scope('CTC_Logits'):
            self.parameters['CTC_Reshape'] = tensorflow.reshape(tensor=self.parameters['CTC_Concat'],
                                                                shape=[-1, 2 * self.hiddenNodules],
                                                                name='CTC_Reshape')
            self.parameters['CTC_Logits_Before'] = tensorflow.layers.dense(inputs=self.parameters['CTC_Reshape'],
                                                                           units=self.numClass + 1, activation=None,
                                                                           name='CTC_Logits_Before')
            # 在这个部分手动加上CTC能够进行自动学习的Null标签。
            self.parameters['CTC_Logits'] = tensorflow.reshape(tensor=self.parameters['CTC_Logits_Before'],
                                                               shape=[self.parameters['BatchSize'],
                                                                      self.parameters['TimeStep'], self.numClass + 1],
                                                               name='CTC_Logits')
            self.parameters['CTC_Logits_TimeMajor'] = tensorflow.transpose(self.parameters['CTC_Logits'],
                                                                           perm=(1, 0, 2),
                                                                           name='CTC_Logits_TimeMajor')

        ###################################################################################################
        # CTC Loss Part
        ###################################################################################################

        with tensorflow.name_scope('CTC_Loss'):
            self.parameters['CTC_Loss'] = tensorflow.nn.ctc_loss(labels=self.labelInput,
                                                                 inputs=self.parameters['CTC_Logits_TimeMajor'],
                                                                 sequence_length=self.seqLenInput)
            self.parameters['CTC_BatchCost'] = tensorflow.reduce_mean(input_tensor=self.parameters['CTC_Loss'],
                                                                      axis=None, name='CTC_BatchCost')
            self.parameters['CTC_Decode'], self.parameters['CTC_Probability'] = tensorflow.nn.ctc_beam_search_decoder(
                inputs=self.parameters['CTC_Logits_TimeMajor'], sequence_length=self.seqLenInput, merge_repeated=False)
        self.parameters['CTC_Train'] = tensorflow.train.MomentumOptimizer(
            learning_rate=learningRate, momentum=0.9).minimize(loss=self.parameters['CTC_BatchCost'])

        ###################################################################################################
        # CRF Logits Pretreatment
        ###################################################################################################

        self.parameters['CRF_Label'] = tensorflow.argmax(input=self.parameters['CTC_Logits'][0:self.numClass], axis=2,
                                                         name='CRF_Label')
        # 总感觉这部分有问题……暂且就这样放着吧

        ###################################################################################################
        # CRF  Pretreatment Part
        ###################################################################################################

        with tensorflow.variable_scope('CRF_BLSTM'):
            self.parameters['CRF_Cell_Forward'] = rnn.BasicLSTMCell(num_units=self.hiddenNodules,
                                                                    name='CRF_Cell_Forward')
            self.parameters['CRF_Cell_Backward'] = rnn.BasicLSTMCell(num_units=self.hiddenNodules,
                                                                     name='CRF_Cell_Backward')
            (self.parameters['CRF_Output_Forward'], self.parameters['CRF_Output_Backward']), _ = \
                tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['CRF_Cell_Forward'],
                                                        cell_bw=self.parameters['CRF_Cell_Backward'],
                                                        inputs=self.dataInput, sequence_length=self.seqLenInput,
                                                        dtype=tensorflow.float32)
            self.parameters['CRF_Concat'] = tensorflow.concat(
                (self.parameters['CRF_Output_Forward'], self.parameters['CRF_Output_Backward']), axis=2)

        ###################################################################################################
        # CRF  Logits Part
        ###################################################################################################

        with tensorflow.name_scope('CRF_Logits'):
            self.parameters['CRF_Reshape'] = tensorflow.reshape(tensor=self.parameters['CRF_Concat'],
                                                                shape=[-1, 2 * self.hiddenNodules], name='CRF_Reshape')
            self.parameters['CRF_Logits_Before'] = tensorflow.layers.dense(inputs=self.parameters['CRF_Reshape'],
                                                                           units=self.numClass, activation=None,
                                                                           name='CRF_Logits_Before')
            self.parameters['CRF_Logits'] = tensorflow.reshape(tensor=self.parameters['CRF_Logits_Before'],
                                                               shape=[self.parameters['BatchSize'],
                                                                      self.parameters['TimeStep'], self.numClass],
                                                               name='CRF_Logits')

        ###################################################################################################
        # CRF  Loss Part
        ###################################################################################################
        with tensorflow.name_scope('CRF_Loss'):
            self.parameters['CRF_LogLikelihood'], self.parameters['CRF_TransitionParams'] = crf.crf_log_likelihood(
                inputs=self.parameters['CRF_Logits'], tag_indices=self.parameters['CRF_Label'],
                sequence_lengths=self.seqLenInput)
            self.parameters['CRF_BatchCost'] = tensorflow.reduce_mean(
                input_tensor=-self.parameters['CRF_LogLikelihood'],
                name='CRF_BatchCost')
        self.parameters['CRF_Train'] = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(
            loss=self.parameters['CRF_BatchCost'])


if __name__ == '__main__':
    classifier = BLSTM_CTC_BLSTM_CRF(trainData=None, trainLabel=None, trainSeqLength=None, featureShape=30, numClass=4)
    print(classifier.information)
