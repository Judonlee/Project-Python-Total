import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy
from __Base.Shuffle import Shuffle_Train


class Test(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, featureShape, numClass, batchSize=32, learningRate=1e-4, startFlag=True,
                 graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.featureShape = featureShape
        self.numClass = numClass
        super(Test, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                   learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                   graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'Just Test'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.X = tensorflow.placeholder(dtype=tensorflow.float32, shape=[2, 3])
        self.Y = tensorflow.placeholder(dtype=tensorflow.float32, shape=None)
        self.cal = tensorflow.multiply(x=self.X, y=self.Y)


if __name__ == '__main__':
    classifier = Test(trainData=None, trainLabel=None, featureShape=None, numClass=None)
    print(classifier.session.run(fetches=classifier.cal,
                                 feed_dict={classifier.X: [[1, 2, 3], [4, 5, 6]],
                                            classifier.Y: [[7, 8, 9], [10, 11, 12]]}))
