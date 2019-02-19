import numpy
import tensorflow
from tensorflow.contrib import rnn
from DepressionRecognition.Model.DBLSTM import DBLSTM
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer
from DepressionRecognition.Loader import Load_DBLSTM


class DBLSTM_Part(DBLSTM):
    def __init__(self, trainData, trainLabel, trainSeq, featureShape=40, firstAttention=None, secondAttention=None,
                 firstAttentionScope=None, secondAttentionScope=None, firstAttentionName=None, secondAttentionName=None,
                 batchSize=32, rnnLayers=2, hiddenNodules=128, learningRate=1E-3, startFlag=True, graphRevealFlag=True,
                 graphPath='logs/', occupyRate=-1):
        super(DBLSTM_Part, self).__init__(
            trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq, featureShape=featureShape,
            firstAttention=firstAttention, secondAttention=secondAttention, firstAttentionScope=firstAttentionScope,
            secondAttentionScope=secondAttentionScope, firstAttentionName=firstAttentionName,
            secondAttentionName=secondAttentionName, batchSize=batchSize, rnnLayers=rnnLayers,
            hiddenNodules=hiddenNodules, learningRate=learningRate, startFlag=startFlag,
            graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=None, name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int64, shape=None, name='seqInput')

        ##########################################################################

        self.parameters['BatchSize'], self.parameters['TimeStep'], _ = tensorflow.unstack(
            tensorflow.shape(input=self.dataInput, name='Parameter'))

        ##########################################################################

        with tensorflow.variable_scope('First_BLSTM'):
            self.parameters['First_FW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)
            self.parameters['First_BW_Cell'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=[rnn.LSTMCell(num_units=self.hiddenNodules) for _ in range(self.rnnLayers)], state_is_tuple=True)

            self.parameters['First_Output'], self.parameters['First_FinalState'] = \
                tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['First_FW_Cell'],
                                                        cell_bw=self.parameters['First_BW_Cell'], inputs=self.dataInput,
                                                        sequence_length=self.seqInput, dtype=tensorflow.float32)

        ##########################################################################

        if self.firstAttention is None:
            self.parameters['First_FinalOutput'] = tensorflow.concat(
                [self.parameters['First_FinalState'][self.rnnLayers - 1][0].h,
                 self.parameters['First_FinalState'][self.rnnLayers - 1][1].h], axis=1)
        else:
            self.firstAttentionList = self.firstAttention(dataInput=self.parameters['First_Output'],
                                                          scopeName=self.firstAttentionName,
                                                          hiddenNoduleNumber=2 * self.hiddenNodules,
                                                          attentionScope=self.firstAttentionScope, blstmFlag=True)

    def Valid(self):
        trainData, trainLabel, trainSeq = self.data, self.label, self.seq

        result = self.session.run(fetches=self.firstAttentionList['FinalResult'],
                                  feed_dict={self.dataInput: trainData[0], self.labelInput: trainLabel[0],
                                             self.seqInput: trainSeq[0]})
        print(result)
        print(numpy.shape(result))


if __name__ == '__main__':
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = Load_DBLSTM()
    classifier = DBLSTM_Part(trainData=trainData, trainLabel=trainLabel, trainSeq=trainSeq,
                             firstAttention=MonotonicAttentionInitializer, secondAttention=None,
                             firstAttentionScope=8, secondAttentionScope=None,
                             firstAttentionName='MA_1', secondAttentionName='MA_2')
    classifier.Valid()
