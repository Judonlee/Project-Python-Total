import tensorflow
from CTC_Project.Module.BaseClass import NeuralNetwork_Base
import numpy
from tensorflow.contrib import rnn


class LSTM_Test(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, rnnLayers=1,
                 batchSize=32, learningRate=0.0001, startFlag=True, graphRevealFlag=True, graphPath='logs/',
                 occupyRate=-1):
        '''
        :param trainLabel:      In this case, trainLabel is the targets.
        :param trainSeqLength:  In this case, the trainSeqLength are needed which is the length of each cases.
        :param featureShape:    designate how many features in one vector.
        :param hiddenNodules:   designite the number of hidden nodules.
        :param rnnLayers:       designate the number of rnn layers.
        :param numClass:        designite the number of classes
        '''

        self.featureShape = featureShape
        self.trainSeqLength = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        self.rnnLayer = rnnLayers
        super(LSTM_Test, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                        learningRate=learningRate, startFlag=startFlag,
                                        graphRevealFlag=graphRevealFlag,
                                        graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This Model uses the Final_Pooling to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, 1], name='dataInput')
        self.depth = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='depth')
        self.cell = rnn.BasicLSTMCell(num_units=1, state_is_tuple=True)
        self.outputs, self.laststate = tensorflow.nn.dynamic_rnn(inputs=self.dataInput, cell=self.cell,
                                                                 dtype=tensorflow.float32, sequence_length=self.depth)
        self.shape = tensorflow.shape(self.laststate)


if __name__ == '__main__':
    trainData = numpy.random.randn(3, 10, 1)
    trainData[1, 6:] = 0
    print(trainData)
    classifier = LSTM_Test(trainData=trainData, trainLabel=None, trainSeqLength=None, featureShape=None,
                           numClass=1)
    print(classifier.information)
    print(classifier.session.run(fetches=classifier.outputs,
                                 feed_dict={classifier.dataInput: trainData, classifier.depth: [10, 6, 10]}))
    print(classifier.session.run(fetches=classifier.laststate,
                                 feed_dict={classifier.dataInput: trainData, classifier.depth: [10, 6, 10]}))
    print(classifier.session.run(fetches=classifier.shape,
                                 feed_dict={classifier.dataInput: trainData, classifier.depth: [10, 6, 10]}))
