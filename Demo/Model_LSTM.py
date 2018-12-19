import tensorflow
import numpy
import random


def Shuffle(data, label, seqLen):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel, newSeqLen = [], [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
        newSeqLen.append(seqLen[sample])
    return newData, newLabel, newSeqLen


class SingleLSTM:
    def __init__(self, trainData, trainLabel, trainSeq, featureShape, batchSize=32, hiddenNodules=128,
                 learningRate=1E-3, startFlag=True, graphRevealFlag=True, graphPath='logs/'):
        '''
        :param trainData:训练数据输入
        :param trainLabel: 训练标签输入
        :param trainSeq: 训练数据的长度输入
        :param featureShape:  特征的长度
        :param batchSize: 一次处理的数据量
        :param learningRate: 学习率
        :param startFlag: 是否进行初始化
        :param graphRevealFlag: 是否保存图片
        :param graphPath: 保存图片的话，保存的路径
        '''
        self.data = trainData
        self.label = trainLabel
        self.seqLen = trainSeq
        self.featureShape = featureShape
        self.batchSize = batchSize
        self.hiddenNodules = hiddenNodules

        # GPU的按需分配策略
        config = tensorflow.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tensorflow.Session(config=config)

        self.parameters = {}  # 保存的网络结构的参数的字典
        self.BuildNetwork(learningRate=learningRate)  # 进行网络结构的搭建

        self.information = '这是基本的LSTM结构，各层的结构如下：'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + '\t' + str(self.parameters[sample])

        if graphRevealFlag:  # 保存整体的路径图
            tensorflow.summary.FileWriter(graphPath, self.session.graph)

        if startFlag:  # 神经网络初始化
            self.session.run(tensorflow.global_variables_initializer())

    def BuildNetwork(self, learningRate):
        '''
        构筑网络结构
        :param learningRate:输入学习率
        :return: 没有输出
        '''
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, numpy.shape(self.label)[1]],
                                                 name='labelInput')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]
        self.parameters['TimeStep'] = tensorflow.shape(input=self.dataInput, name='TimeStep')[1]

        self.parameters['LSTM_Cell'] = tensorflow.nn.rnn_cell.LSTMCell(
            num_units=self.hiddenNodules, state_is_tuple=True, name='RNN_Cell')
        self.parameters['LSTM_Result'], self.parameters['LSTM_FinalState'] = tensorflow.nn.dynamic_rnn(
            cell=self.parameters['LSTM_Cell'], inputs=self.dataInput, sequence_length=self.seqLenInput,
            dtype=tensorflow.float32)

        self.parameters['Logits'] = tensorflow.layers.dense(
            inputs=self.parameters['LSTM_FinalState'].h, units=numpy.shape(self.label)[1], activation=None,
            name='Logits')
        self.parameters['Loss'] = tensorflow.losses.softmax_cross_entropy(
            onehot_labels=self.labelInput, logits=self.parameters['Logits'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        '''
        进行训练，没有参数需要输入
        :return: 没有输出
        '''
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)

        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchData = []
            batchLabel = trainLabel[startPosition:startPosition + self.batchSize]
            batchSeq = trainSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(trainSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                currentData = numpy.concatenate(
                    (trainData[index], numpy.zeros((maxLen - len(trainData[index]), len(trainData[index][0])))), axis=0)
                batchData.append(currentData)

            # print(numpy.shape(batchData), numpy.shape(batchLabel), numpy.shape(batchSeq))

            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                  self.seqLenInput: batchSeq})
            # print(loss)
            totalLoss += loss

            output = '\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss)
            print(output, end='')
            startPosition += self.batchSize
        return totalLoss

    def Test(self, testData, testLabel, testSeq):
        '''
        进行测试，返回混淆矩阵
        :param testData: 测试用的数据
        :param testLabel: 测试的标签
        :param testSeq: 测试样本的长度
        :return: 混淆矩阵的返回
        '''
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            batchData = []
            batchSeq = testSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(testSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(testData))):
                currentData = numpy.concatenate(
                    (testData[index], numpy.zeros((maxLen - len(testData[index]), len(testData[index][0])))), axis=0)
                batchData.append(currentData)

            predict = self.session.run(fetches=self.parameters['Logits'],
                                       feed_dict={self.dataInput: batchData, self.seqLenInput: batchSeq})
            totalPredict.extend(predict)
            startPosition += self.batchSize
            print('\rTesting %d/%d' % (startPosition, len(testData)), end='')

        confusionMatrix = numpy.ones((numpy.shape(testLabel)[1], numpy.shape(testLabel)[1]))
        for index in range(len(totalPredict)):
            confusionMatrix[numpy.argmax(numpy.array(testLabel[index]))][
                numpy.argmax(numpy.array(totalPredict[index]))] += 1
        return confusionMatrix

    def Save(self, savepath):
        '''
        保存神经网络的参数，目前应该不会用到，不过还是写上去了
        :param savepath: 保存的路径
        :return: 没有输出
        '''
        saver = tensorflow.train.Saver()
        saver.save(self.session, savepath)

    def Load(self, loadpath):
        '''
        读取神经网络参数，目前应该不会用到，不过还是写上去了
        :param loadpath: 读取的路径
        :return: 没有输出
        '''
        saver = tensorflow.train.Saver()
        saver.restore(self.session, loadpath)

    def SaveGraph(self, graphPath):
        '''
        保存整体的示意图，目前应该不会用到，不过还是写上去了
        :param graphPath: 保存图片的路径
        :return: 没有输出
        '''
        tensorflow.summary.FileWriter(graphPath, self.session.graph)
