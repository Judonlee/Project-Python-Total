import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import numpy
from MultiModalTest.Loader.IEMOCAP_Loader import CNNLoaderLeaveOneSpeaker


class CNN_Origin(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, XShape, YShape, numClass, learningRate=1E-3, hiddenNodules=128,
                 batchSize=32, startFlag=True, graphRevealFlag=False, graphPath='logs/', occupyRate=-1):
        self.xShape = XShape
        self.yShape = YShape
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        super(CNN_Origin, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                         learningRate=learningRate, startFlag=startFlag,
                                         graphRevealFlag=graphRevealFlag, graphPath=graphPath, occupyRate=occupyRate)
        self.information = ''
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.xShape, self.yShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.numClass],
                                                 name='labelInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]
        self.parameters['DataInputAddAxis'] = self.dataInput[:, :, :, tensorflow.newaxis]

        ##########################################################################################

        self.parameters['Layer1st_Conv'] = tensorflow.layers.conv2d(
            inputs=self.parameters['DataInputAddAxis'], filters=64, kernel_size=[8, 8], padding='SAME',
            activation=tensorflow.nn.relu, name='Layer1st_Conv')
        self.parameters['Layer1st_BatchNormalization'] = tensorflow.nn.batch_normalization(
            x=self.parameters['Layer1st_Conv'], )

    def MediaTest(self):
        trainData = self.data[0:self.batchSize]
        trainLabel = self.label[0:self.batchSize]
        result = self.session.run(fetches=self.parameters['Layer1st_Conv'],
                                  feed_dict={self.dataInput: trainData, self.labelInput: trainLabel})
        print(result)
        print(numpy.shape(result))


if __name__ == '__main__':
    trainData, trainLabel, testData, testLabel = CNNLoaderLeaveOneSpeaker(
        loadpath='E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands30-CNN/', appointSession=1,
        appointGender='Female')
    classifier = CNN_Origin(trainData=trainData, trainLabel=trainLabel, XShape=500, YShape=30, numClass=4)
    classifier.MediaTest()
