import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
from tensorflow.contrib import rnn
from __Base.Shuffle import Shuffle
import numpy


class E2E_FinalPooling(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, numClass, hiddenNodules=128, rnnLayers=1,
                 batchSize=64, learningRate=1e-4, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        self.seqLen = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        self.rnnLayer = rnnLayers
        super(E2E_FinalPooling, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                               learningRate=learningRate, startFlag=startFlag,
                                               graphRevealFlag=graphRevealFlag, graphPath=graphPath,
                                               occupyRate=occupyRate)

        self.information = 'This Model uses the CTC and Final_Pooling to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, 1], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.numClass],
                                                 name='labelInput')
        self.dropOutRate = tensorflow.placeholder(dtype=tensorflow.float32, name='dropOutRate')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, name='seqLenInput')

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv1d(inputs=self.dataInput, filters=8, kernel_size=64,
                                                                    padding='SAME', name='Layer1st_Conv')
        self.parameters['Layer1st_MaxPooling'] = tensorflow.layers.max_pooling1d(
            inputs=self.parameters['Layer1st_Conv'], pool_size=10, strides=10, padding='SAME',
            name='Layer1st_MaxPooling')
        self.parameters['Layer1st_DropOut'] = tensorflow.nn.dropout(x=self.parameters['Layer1st_MaxPooling'],
                                                                    keep_prob=self.dropOutRate, name='Layer1st_DropOut')

        self.parameters['Layer2nd_Conv'] = tensorflow.layers.conv1d(inputs=self.parameters['Layer1st_DropOut'],
                                                                    filters=6, kernel_size=128, padding='SAME',
                                                                    name='Layer2nd_Conv')
        self.parameters['Layer2nd_MaxPooling'] = tensorflow.layers.max_pooling1d(
            inputs=self.parameters['Layer2nd_Conv'], pool_size=8, strides=8, padding='SAME', name='Layer2nd_MaxPooling')
        self.parameters['Layer2nd_DropOut'] = tensorflow.nn.dropout(x=self.parameters['Layer2nd_MaxPooling'],
                                                                    keep_prob=self.dropOutRate, name='Layer2nd_DropOut')

        self.parameters['Layer3rd_Conv'] = tensorflow.layers.conv1d(inputs=self.parameters['Layer2nd_DropOut'],
                                                                    filters=6, kernel_size=256, padding='SAME',
                                                                    name='Layer3rd_Conv')
        self.parameters['Layer3rd_MaxPooling'] = tensorflow.layers.max_pooling1d(
            inputs=self.parameters['Layer3rd_Conv'], pool_size=8, strides=8, padding='SAME', name='Layer3rd_MaxPooling')
        self.parameters['Layer3rd_DropOut'] = tensorflow.nn.dropout(x=self.parameters['Layer3rd_MaxPooling'],
                                                                    keep_prob=self.dropOutRate, name='Layer3rd_DropOut')

        ############################################################################################
        # BLSTM Part
        ############################################################################################

        self.parameters['Layer4th_RNN_Cell_Forward'] = rnn.BasicLSTMCell(num_units=self.hiddenNodules,
                                                                         name='Layer4th_RNN_Cell_Forward')
        self.parameters['Layer4th_RNN_Cell_Backward'] = rnn.BasicLSTMCell(num_units=self.hiddenNodules,
                                                                          name='Layer4th_RNN_Cell_Backward')
        (self.parameters['Layer4th_RNN_Output_Forward'], self.parameters['Layer4th_RNN_Output_Backward']), _ = \
            tensorflow.nn.bidirectional_dynamic_rnn(cell_fw=self.parameters['Layer4th_RNN_Cell_Forward'],
                                                    cell_bw=self.parameters['Layer4th_RNN_Cell_Backward'],
                                                    inputs=self.parameters['Layer3rd_DropOut'],
                                                    sequence_length=self.seqLenInput, dtype=tensorflow.float32)
        self.parameters['Layer4th_RNN_Concat'] = tensorflow.concat(
            (self.parameters['Layer4th_RNN_Output_Forward'], self.parameters['Layer4th_RNN_Output_Backward']), axis=2)
        self.parameters['Layer4th_Result'] = tensorflow.reshape(tensor=self.parameters['Layer4th_RNN_Concat'][:, -1, :],
                                                                shape=[-1, 256], name='Layer4th_Result')
        self.parameters['Layer5th_Logits'] = tensorflow.layers.dense(inputs=self.parameters['Layer4th_Result'],
                                                                     units=self.numClass, activation=None,
                                                                     name='Layer5th_Logits')
        self.parameters['Loss'] = tensorflow.reduce_mean(tensorflow.losses.softmax_cross_entropy(
            onehot_labels=self.labelInput, logits=self.parameters['Layer5th_Logits']))
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)
        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchLabel = trainLabel[startPosition:startPosition + self.batchSize]
            batchSeq = trainSeq[startPosition:startPosition + self.batchSize]
            batchData = []

            dataShape = 0
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                if len(trainData[index]) > dataShape: dataShape = len(trainData[index])
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                currentData = numpy.concatenate(
                    (trainData[index], numpy.zeros(dataShape - len(trainData[index]))), axis=0)
                batchData.append(currentData)
            batchData = numpy.array(batchData)
            batchData = batchData[:, :, numpy.newaxis]
            # print(numpy.shape(batchData))

            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                  self.seqLenInput: batchSeq, self.dropOutRate: 0.5})
            print('\rBatch %d/%d: Loss = %f' % (startPosition, len(trainData), loss), end='')
            totalLoss += loss
            startPosition += self.batchSize
        return totalLoss


if __name__ == '__main__':
    classifier = E2E_FinalPooling(trainData=None, trainLabel=None, trainSeqLength=None, numClass=4)
    print(classifier.information)
