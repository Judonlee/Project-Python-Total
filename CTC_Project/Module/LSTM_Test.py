import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy


class LSTM_Test(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, numClass, batchSize=32, learningRate=0.0001,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        '''
        :param trainLabel:      In this case, trainLabel is the targets.
        :param trainSeqLength:  In this case, the trainSeqLength are needed which is the length of each cases.
        :param featureShape:    designate how many features in one vector.
        :param hiddenNodules:   designite the number of hidden nodules.
        :param rnnLayers:       designate the number of rnn layers.
        :param numClass:        designite the number of classes
        '''

        self.trainSeqLength = trainSeqLength
        self.numClass = numClass
        super(LSTM_Test, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                        learningRate=learningRate, startFlag=startFlag,
                                        graphRevealFlag=graphRevealFlag,
                                        graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This Model uses the Final_Pooling to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, None], name='dataInput')
        self.labelInput = tensorflow.sparse_placeholder(dtype=tensorflow.int32, shape=None, name='labelInput')
        self.seqInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqInput')

        self.dataTimeMajor = tensorflow.transpose(self.dataInput, perm=(1, 0, 2))

        self.dense = tensorflow.sparse_tensor_to_dense(sp_input=self.labelInput)
        self.cost = tensorflow.nn.ctc_loss(labels=self.labelInput, inputs=self.dataTimeMajor,
                                           sequence_length=self.seqInput)
        self.decode = tensorflow.nn.ctc_greedy_decoder(inputs=self.dataTimeMajor, sequence_length=self.seqInput,
                                                       merge_repeated=False)
        self.decodeDense = tensorflow.sparse_tensor_to_dense(sp_input=self.decode[0][0])


if __name__ == '__main__':
    trainData = [[[-10, 1, -10, -10], [-10, -10, 1, -10], [-10, -10, 1, -10]],
                 [[-10, 1, -10, -10], [-10, -10, 1, -10], [-10, -10, 1, -10]]]
    print(numpy.shape(trainData))
    trainLabel = ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2]], [1, 2, 3, 1, 2, 3], [2, 3])
    print(trainData)
    classifier = LSTM_Test(trainData=trainData, trainLabel=trainLabel, trainSeqLength=None, numClass=1)
    print(classifier.information)
    print(classifier.session.run(fetches=classifier.dataTimeMajor, feed_dict={classifier.dataInput: trainData}))
    print(classifier.session.run(fetches=classifier.dense, feed_dict={classifier.labelInput: trainLabel}))
    print(numpy.shape(
        classifier.session.run(fetches=classifier.dataTimeMajor, feed_dict={classifier.dataInput: trainData})))
    print(classifier.session.run(fetches=classifier.cost,
                                 feed_dict={classifier.dataInput: trainData, classifier.labelInput: trainLabel,
                                            classifier.seqInput: [3, 3]}))
    print(classifier.session.run(fetches=classifier.decodeDense,
                                 feed_dict={classifier.dataInput: trainData, classifier.labelInput: trainLabel,
                                            classifier.seqInput: [3, 3]}))
