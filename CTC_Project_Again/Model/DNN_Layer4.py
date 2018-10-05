import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy
from __Base.Shuffle import Shuffle_Train


class DNN(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, featureShape, numClass, batchSize=32, learningRate=1e-4, startFlag=True,
                 graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        '''
        :param trainLabel:      In this case, trainLabel is the targets.
        :param trainSeqLength:  In this case, the trainSeqLength are needed which is the length of each cases.
        :param featureShape:    designate how many features in one vector.
        :param hiddenNodules:   designite the number of hidden nodules.
        :param rnnLayers:       designate the number of rnn layers.
        :param numClass:        designite the number of classes
        '''

        self.featureShape = featureShape
        self.numClass = numClass
        super(DNN, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                  learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                  graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This Model uses the DNN-Layer4 to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.numClass],
                                                 name='labelInput')

        self.parameters['Layer1st_FC'] = tensorflow.layers.dense(inputs=self.dataInput, units=256,
                                                                 activation=tensorflow.nn.relu, name='Layer1_FC')
        self.parameters['Layer2nd_FC'] = tensorflow.layers.dense(inputs=self.parameters['Layer1st_FC'], units=256,
                                                                 activation=tensorflow.nn.relu, name='Layer2nd_FC')
        self.parameters['Layer3rd_FC'] = tensorflow.layers.dense(inputs=self.parameters['Layer2nd_FC'], units=256,
                                                                 activation=tensorflow.nn.relu, name='Layer3rd_FC')
        self.parameters['Layer4th_FC'] = tensorflow.layers.dense(inputs=self.parameters['Layer3rd_FC'], units=256,
                                                                 activation=tensorflow.nn.relu, name='Layer4th_FC')
        self.parameters['Logits'] = tensorflow.layers.dense(inputs=self.parameters['Layer4th_FC'], units=self.numClass,
                                                            activation=None, name='Predict')
        self.parameters['BatchLoss'] = tensorflow.nn.softmax_cross_entropy_with_logits(labels=self.labelInput,
                                                                                       logits=self.parameters['Logits'],
                                                                                       name='BatchLoss')
        self.parameters['Loss'] = tensorflow.reduce_mean(input_tensor=self.parameters['BatchLoss'], name='Loss')
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(loss=self.parameters['Loss'])

    def Train(self):
        startPosition = 0
        trainData, trainLabel = Shuffle_Train(data=self.data, label=self.label)
        totalLoss = 0
        while startPosition < len(trainData):
            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train], feed_dict={
                self.dataInput: trainData[startPosition:startPosition + self.batchSize],
                self.labelInput: trainLabel[startPosition:startPosition + self.batchSize]})
            print('\rTraining %d/%d : Loss = %f' % (startPosition, len(trainData), loss), end='')
            startPosition += self.batchSize
            totalLoss += loss
        print()
        return totalLoss

    def Test(self, testData, testLabel):
        startPosition = 0
        totalPredict = []
        while startPosition < len(testData):
            batchPredict = self.session.run(fetches=self.parameters['Logits'], feed_dict={
                self.dataInput: testData[startPosition:startPosition + self.batchSize]})
            startPosition += self.batchSize
            totalPredict.extend(batchPredict)

        matrix = numpy.zeros((self.numClass, self.numClass))
        for index in range(len(testLabel)):
            matrix[numpy.argmax(numpy.array(testLabel[index]))][numpy.argmax(numpy.array(totalPredict[index]))] += 1
        #print()
        #print(matrix)
        return matrix


if __name__ == '__main__':
    classifier = DNN(trainData=None, trainLabel=None, featureShape=30, numClass=4)
    print(classifier.information)
