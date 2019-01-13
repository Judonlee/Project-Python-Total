import tensorflow
from MultiModalTest.Model.CTC_BLSTM_LC_Attention import CTC_LC_Attention
import numpy
from __Base.Shuffle import Shuffle


class CTC_LA_Transform(CTC_LC_Attention):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, emotionClass, phonemeClass, rnnLayers,
                 attentionScope, punishmentDegree, hiddenNodules=128, batchSize=32, startFlag=True,
                 graphRevealFlag=False, graphPath='logs/', occupyRate=-1, initialParameterPath=''):
        self.emotionClass = emotionClass
        self.punishmentDegree = punishmentDegree
        super(CTC_LA_Transform, self).__init__(
            trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeqLength, featureShape=featureShape,
            numClass=phonemeClass, rnnLayers=rnnLayers, attentionScope=attentionScope, hiddenNodules=hiddenNodules,
            batchSize=batchSize, startFlag=False, graphRevealFlag=False, graphPath=None, occupyRate=occupyRate)

        self.AppendNetwork()

        if startFlag: self.session.run(tensorflow.global_variables_initializer())
        if startFlag and initialParameterPath != '': self.LoadPart(loadpath=initialParameterPath)
        if not startFlag and initialParameterPath != '': self.Load(loadpath=initialParameterPath)

        ###################################################################

        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + '\t' + str(self.parameters[sample])
        print(self.information)

        ###################################################################

        if graphRevealFlag:
            tensorflow.summary.FileWriter(graphPath, self.session.graph)

        ###################################################################

    def AppendNetwork(self):
        self.punishmentInput = tensorflow.placeholder(dtype=tensorflow.float32, name='punishmentInput')

        with tensorflow.variable_scope('SpeechEmotion'):
            self.parameters['Emotion_RNN_Cell_Forward'] = []
            self.parameters['Emotion_RNN_Cell_Backward'] = []

            for layer in range(self.rnnLayers):
                self.parameters['Emotion_RNN_Cell_Forward'].append(
                    tensorflow.nn.rnn_cell.LSTMCell(num_units=self.hiddenNodules, state_is_tuple=True,
                                                    name='Emotion_RNN_Cell_Forward_%d' % layer))
                self.parameters['Emotion_RNN_Cell_Backward'].append(
                    tensorflow.nn.rnn_cell.LSTMCell(num_units=self.hiddenNodules, state_is_tuple=True,
                                                    name='Emotion_RNN_Cell_Backward_%d' % layer))

            self.parameters['Emotion_Layer_Forward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=self.parameters['Emotion_RNN_Cell_Forward'], state_is_tuple=True)
            self.parameters['Emotion_Layer_Backward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
                cells=self.parameters['Emotion_RNN_Cell_Backward'], state_is_tuple=True)

            (self.parameters['Emotion_RNN_OutputForward'], self.parameters['Emotion_RNN_OutputBackward']), _ = \
                tensorflow.nn.bidirectional_dynamic_rnn(
                    cell_fw=self.parameters['Emotion_Layer_Forward'], cell_bw=self.parameters['Emotion_Layer_Backward'],
                    inputs=self.dataInput, sequence_length=self.seqLenInput, dtype=tensorflow.float32)

        self.parameters['Emotion_RNN_Concat'] = tensorflow.concat(
            [self.parameters['Emotion_RNN_OutputForward'], self.parameters['Emotion_RNN_OutputBackward']], axis=2)
        self.parameters['Emotion_RNN_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['Emotion_RNN_Concat'], shape=[-1, 2 * self.hiddenNodules],
            name='Emotion_RNN_Reshape')

        self.parameters['Emotion_Attention_Value'] = tensorflow.layers.dense(
            inputs=self.parameters['Emotion_RNN_Reshape'], units=1, activation=tensorflow.nn.tanh,
            name='Emotion_Attention_Value')

        with tensorflow.variable_scope('Emotion_Attention_Concat'):
            self.parameters['Emotion_Attention_Concat'] = tensorflow.concat(
                [self.parameters['Emotion_Attention_Value'][0:-self.attentionScope],
                 self.parameters['Emotion_Attention_Value'][1:-self.attentionScope + 1]], axis=1)
            for concatCounter in range(2, self.attentionScope):
                self.parameters['Emotion_Attention_Concat'] = tensorflow.concat(
                    [self.parameters['Emotion_Attention_Concat'],
                     self.parameters['Emotion_Attention_Value'][concatCounter:-self.attentionScope + concatCounter]],
                    axis=1)

        self.parameters['Emotion_Attention_Evaluation'] = tensorflow.nn.softmax(
            logits=self.parameters['Emotion_Attention_Concat'], name='Emotion_Attention_Evaluation')

        with tensorflow.variable_scope('Emotion_AttentionAdd'):
            self.parameters['Emotion_MultiPart_%04d' % 0] = tensorflow.tile(
                input=self.parameters['Emotion_Attention_Evaluation'][:, 0:1], multiples=[1, 2 * self.hiddenNodules],
                name='Emotion_MultiPart_%04d' % 0)
            self.parameters['Emotion_RNN_WithAttention_Total'] = tensorflow.multiply(
                x=self.parameters['Emotion_RNN_Reshape'][0:-self.attentionScope],
                y=self.parameters['Emotion_MultiPart_%04d' % 0])

            for MultiCounter in range(1, self.attentionScope):
                self.parameters['Emotion_MultiPart_%04d' % MultiCounter] = tensorflow.tile(
                    input=self.parameters['Emotion_Attention_Evaluation'][:, MultiCounter:MultiCounter + 1],
                    multiples=[1, 2 * self.hiddenNodules], name='Emotion_MultiPart_%04d' % MultiCounter)
                self.parameters['Emotion_RNN_WithAttention_%04d' % MultiCounter] = tensorflow.multiply(
                    x=self.parameters['Emotion_RNN_Reshape'][MultiCounter:-self.attentionScope + MultiCounter],
                    y=self.parameters['Emotion_MultiPart_%04d' % MultiCounter])

                self.parameters['Emotion_RNN_WithAttention_Total'] = tensorflow.add(
                    x=self.parameters['Emotion_RNN_WithAttention_Total'],
                    y=self.parameters['Emotion_RNN_WithAttention_%04d' % MultiCounter])

        self.parameters['Emotion_RNN_Final'] = tensorflow.concat(
            [self.parameters['Emotion_RNN_WithAttention_Total'],
             tensorflow.zeros(shape=[self.attentionScope, 2 * self.hiddenNodules])], axis=0)
        self.parameters['Emotion_Logits'] = tensorflow.layers.dense(
            inputs=self.parameters['Emotion_RNN_Final'], units=self.emotionClass, activation=None,
            name='Emotion_Logits')
        self.parameters['Emotion_Logits_Reshape'] = tensorflow.reshape(
            tensor=self.parameters['Emotion_Logits'],
            shape=[self.parameters['BatchSize'], self.parameters['TimeStep'], self.emotionClass],
            name='Emotion_Logits_Reshape')

        self.parameters['Emotion_Logits_TimeMajor'] = tensorflow.transpose(
            self.parameters['Emotion_Logits_Reshape'], perm=[1, 0, 2], name='Emotion_Logits_TimeMajor')

        ############################################################################

        with tensorflow.variable_scope('Emotion_Cost'):
            self.parameters['Emotion_CTC_Loss'] = tensorflow.nn.ctc_loss(
                labels=self.labelInput, inputs=self.parameters['Emotion_Logits_TimeMajor'],
                sequence_length=self.seqLenInput, ignore_longer_outputs_than_inputs=True)

            self.parameters['Emotion_PunishmentLoss'] = self.punishmentInput * tensorflow.reduce_mean(
                tensorflow.abs(self.parameters['Emotion_Attention_Value'] - self.parameters['Attention_Value']))

            self.parameters['Emotion_TotalCost'] = tensorflow.reduce_mean(self.parameters['Emotion_CTC_Loss']) + \
                                                   self.parameters['Emotion_PunishmentLoss']

        self.EmotionTrain = tensorflow.train.AdamOptimizer(learning_rate=self.learningRate).minimize(
            self.parameters['Emotion_TotalCost'], var_list=tensorflow.global_variables()[36:])
        self.decode, self.logProbability = tensorflow.nn.ctc_beam_search_decoder(
            inputs=self.parameters['Emotion_Logits_TimeMajor'], sequence_length=self.seqLenInput, merge_repeated=False)
        self.decodeDense = tensorflow.sparse_tensor_to_dense(sp_input=self.decode[0])

    def LoadPart(self, loadpath):
        saver = tensorflow.train.Saver(var_list=tensorflow.global_variables()[0:36])
        saver.restore(self.session, loadpath)

    def EmotionTrainEpisode(self, learningRate):
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)

        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchData = []
            batachSeq = trainSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(trainSeq[startPosition:startPosition + self.batchSize]) + self.attentionScope
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

            loss, ctcLoss, punishmentLoss, _ = self.session.run(
                fetches=[self.parameters['Emotion_TotalCost'], self.parameters['Emotion_CTC_Loss'],
                         self.parameters['Emotion_PunishmentLoss'], self.EmotionTrain],
                feed_dict={self.dataInput: batchData, self.labelInput: (indices, values, shape),
                           self.seqLenInput: batachSeq, self.learningRate: learningRate,
                           self.punishmentInput: self.punishmentDegree})
            totalLoss += loss

            ctcLoss = numpy.average(ctcLoss)

            output = '\rBatch : %d/%d \t Loss : %f\tCTC : %f\tPunishment : %f                                                 ' % (
                startPosition, len(trainData), loss, ctcLoss, punishmentLoss)
            print(output, end='')
            startPosition += self.batchSize
        return totalLoss
