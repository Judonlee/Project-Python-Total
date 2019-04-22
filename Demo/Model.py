import tensorflow
import numpy
import random
from Demo.Attention import StandardAttentionInitializer
from tensorflow.contrib import rnn


def Shuffle(data, label):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel = [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
    return newData, newLabel


class NetworkStructure:
    def __init__(self, trainData, trainLabel, developData, developLabel, rnnLayers=2, hiddenNoduleNumber=128,
                 batchSize=32):
        self.trainData, self.trainLabel, self.developData, self.developLabel = trainData, trainLabel, developData, developLabel
        self.rnnLayers, self.hiddenNoduleNumber = rnnLayers, hiddenNoduleNumber
        self.batchSize = batchSize
        ###########################################################################

        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tensorflow.Session(config=config)
        self.BuildNetwork()
        self.session.run(tensorflow.global_variables_initializer())

    def BuildNetwork(self):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 500, 40], name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, 4], name='labelInput')

        self.parameter = {}
        self.parameter['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.dataInput[:, :, :, tensorflow.newaxis], filters=64, kernel_size=8, strides=1, padding='SAME',
            activation=tensorflow.nn.relu, name='Layer1st_Conv')
        self.parameter['Layer1st_Pooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameter['Layer1st_Conv'], pool_size=2, strides=2, name='Layer1st_Pooling')

        self.parameter['Layer2nd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameter['Layer1st_Pooling'], filters=128, kernel_size=5, strides=1, padding='SAME',
            activation=tensorflow.nn.relu, name='Layer2nd_Conv')
        self.parameter['Layer2nd_Pooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameter['Layer2nd_Conv'], pool_size=2, strides=2, name='Layer2nd_Pooling')

        self.parameter['Layer3rd_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameter['Layer2nd_Pooling'], filters=256, kernel_size=3, strides=1, padding='SAME',
            activation=tensorflow.nn.relu, name='Layer3rd_Conv')
        self.parameter['Layer3rd_Pooling'] = tensorflow.layers.max_pooling2d(
            inputs=self.parameter['Layer3rd_Conv'], pool_size=2, strides=2, name='Layer3rd_Pooling')

        self.parameter['Layer4th_Reshape'] = tensorflow.reshape(tensor=self.parameter['Layer3rd_Pooling'],
                                                                shape=[-1, 62, 1280], name='Layer4th_Reshape')

        ###########################################################################
        # Up Way

        self.parameter['Cell_Forward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNoduleNumber) for _ in range(self.rnnLayers)], state_is_tuple=True)
        self.parameter['Cell_Backward'] = tensorflow.nn.rnn_cell.MultiRNNCell(
            cells=[rnn.LSTMCell(num_units=self.hiddenNoduleNumber) for _ in range(self.rnnLayers)], state_is_tuple=True)

        self.parameter['BLSTM_Output'], self.parameter['BLSTM_FinalState'] = \
            tensorflow.nn.bidirectional_dynamic_rnn(
                cell_fw=self.parameter['Cell_Forward'], cell_bw=self.parameter['Cell_Backward'],
                inputs=self.parameter['Layer4th_Reshape'], dtype=tensorflow.float32)

        self.parameter['AttentionList'] = StandardAttentionInitializer(
            dataInput=self.parameter['BLSTM_Output'], scopeName='StandardAttention',
            hiddenNoduleNumber=2 * self.hiddenNoduleNumber, blstmFlag=True)
        self.parameter['Layer5th_Up'] = self.parameter['AttentionList']['FinalResult']

        ###########################################################################
        # Down Way
        self.parameter['Layer5th_Down'] = tensorflow.reduce_max(input_tensor=self.parameter['Layer4th_Reshape'], axis=1,
                                                                name='Layer5th_Down')

        ###########################################################################
        # Down Way
        self.parameter['Layer5th_Concat'] = tensorflow.concat(
            [self.parameter['Layer5th_Up'], self.parameter['Layer5th_Down']], axis=1, name='Layer5th_Concat')

        self.parameter['FinalPredict'] = tensorflow.layers.dense(inputs=self.parameter['Layer5th_Concat'], units=4,
                                                                 activation=None, name='FinalPredict')

        self.parameter['Loss'] = tensorflow.losses.softmax_cross_entropy(
            onehot_labels=self.labelInput, logits=self.parameter['FinalPredict'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=1E-3).minimize(self.parameter['Loss'])

    def Train(self):
        trainData, trainLabel = Shuffle(data=self.trainData, label=self.trainLabel)

        startPosition = 0
        totalLoss = 0.0
        while startPosition < len(trainData):
            batchData, batchLabel = trainData[startPosition:startPosition + self.batchSize], \
                                    trainLabel[startPosition:startPosition + self.batchSize]
            loss, _ = self.session.run(fetches=[self.parameter['Loss'], self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel})
            totalLoss += loss
            print('\rTraining %d/%d Loss = %f' % (startPosition, len(trainData), loss), end='')
            startPosition += self.batchSize
        return totalLoss

    def Test(self, logname, testData, testLabel):
        startPosition = 0

        predictList = []
        while startPosition < len(testData):
            batchData, batchLabel = testData[startPosition:startPosition + self.batchSize], \
                                    testLabel[startPosition:startPosition + self.batchSize]
            predict = self.session.run(fetches=self.parameter['FinalPredict'],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel})
            predictList.extend(predict)
            startPosition += self.batchSize

        confusionMatrix = numpy.zeros([4, 4])
        for index in range(len(testLabel)):
            confusionMatrix[numpy.argmax(numpy.array(testLabel[index]))][
                numpy.argmax(numpy.array(predictList[index]))] += 1

        with open(logname, 'w') as file:
            for indexX in range(len(confusionMatrix)):
                for indexY in range(len(confusionMatrix[indexX])):
                    if indexY != 0: file.write(',')
                    file.write(str(confusionMatrix[indexX][indexY]))
                file.write('\n')

    def Valid(self):
        batchData, batchLabel = self.developData[0:self.batchSize], self.developLabel[0:self.batchSize]
        print(numpy.shape(batchData), numpy.shape(batchLabel))

        predict = self.session.run(fetches=self.parameter['FinalPredict'],
                                   feed_dict={self.dataInput: batchData, self.labelInput: batchLabel})
        print(numpy.shape(predict))
