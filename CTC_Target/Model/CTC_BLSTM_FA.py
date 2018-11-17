import tensorflow
from CTC_Target.Model.CTC_Multi_BLSTM import CTC_Multi_BLSTM
from CTC_Target.Loader.IEMOCAP_Loader import Load
import numpy


class CTC_Multi_FA(CTC_Multi_BLSTM):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, rnnLayers, hiddenNodules=128,
                 batchSize=32, learningRate=1e-3, startFlag=True, graphRevealFlag=False, graphPath='logs/',
                 occupyRate=-1):
        super(CTC_Multi_FA, self).__init__(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeqLength,
                                           featureShape=featureShape, numClass=numClass, rnnLayers=rnnLayers,
                                           hiddenNodules=hiddenNodules, batchSize=batchSize, learningRate=learningRate,
                                           startFlag=startFlag, graphRevealFlag=graphRevealFlag, graphPath=graphPath,
                                           occupyRate=occupyRate)
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
        # Attention
        ###################################################################################################

        self.parameters['RNN_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['RNN_Concat'],
            shape=[self.parameters['BatchSize'] * self.parameters['TimeStep'], 2 * self.hiddenNodules],
            name='RNN_Reshape')
        self.parameters['FL_Attention'] = tensorflow.layers.dense(
            inputs=self.parameters['RNN_Reshape'], units=2 * self.hiddenNodules, activation=tensorflow.nn.softmax,
            name='FL_Attention')
        self.parameters['FL_Weights'] = tensorflow.tile(
            input=tensorflow.constant(value=[self.hiddenNodules * 2], dtype=tensorflow.float32, shape=[1, 1]),
            multiples=[self.parameters['BatchSize'] * self.parameters['TimeStep'], 2 * self.hiddenNodules],
            name='FL_Weights')
        self.parameters['Attention'] = tensorflow.multiply(x=self.parameters['FL_Attention'],
                                                           y=self.parameters['FL_Weights'], name='Attention')
        self.parameters['RNNWithAttention'] = tensorflow.multiply(x=self.parameters['RNN_Reshape'],
                                                                  y=self.parameters['Attention'],
                                                                  name='RNNWithAttention')

        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['RNNWithAttention'],
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
        self.train = tensorflow.train.RMSPropOptimizer(learning_rate=learningRate).minimize(
            self.parameters['Cost'])
        self.decode, self.logProbability = tensorflow.nn.ctc_beam_search_decoder(
            inputs=self.parameters['Logits_TimeMajor'], sequence_length=self.seqLenInput, merge_repeated=False)
        self.decodeDense = tensorflow.sparse_tensor_to_dense(sp_input=self.decode[0])
    #
    # def Valid(self):
    #     startPosition = 0
    #     while startPosition < len(trainData):
    #         batchData = []
    #         batachSeq = trainSeq[startPosition:startPosition + self.batchSize]
    #
    #         maxLen = max(trainSeq[startPosition:startPosition + self.batchSize])
    #         for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
    #             currentData = numpy.concatenate(
    #                 (trainData[index], numpy.zeros((maxLen - len(trainData[index]), len(trainData[index][0])))), axis=0)
    #             batchData.append(currentData)
    #
    #         result = self.session.run(fetches=self.parameters['LogitsWithAttention'],
    #                                   feed_dict={self.dataInput: batchData, self.seqLenInput: batachSeq})
    #         print(numpy.shape(result))
    #         # print(result[0:5][0:5])
    #         exit()


if __name__ == '__main__':
    bands = 30
    loadpath = 'D:/ProjectData/CTC_Target/Features/Bands%d/' % bands
    trainData, trainLabel, trainSeq, trainScription, testData, testlabel, testSeq, testScription = Load(
        loadpath=loadpath, appoint=1)

    classifier = CTC_Multi_FA(trainData=testData, trainLabel=testlabel, trainSeqLength=testSeq, featureShape=30,
                              numClass=5, rnnLayers=2, graphRevealFlag=True)
    print(classifier.information)
    # classifier.Valid()
    classifier.Train()
