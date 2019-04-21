import tensorflow
import numpy
import random


def Shuffle(data, label):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel = [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
    return newData, newLabel


class NetworkStructure:
    def __init__(self, trainData, trainLabel, developData, developLabel, batchSize=32):
        self.trainData, self.trainLabel, self.developData, self.developLabel = trainData, trainLabel, developData, developLabel
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

    def Train(self):
        pass

    def Test(self, testData, testLabel):
        pass

    def Valid(self):
        batchData, batchLabel = self.developData[0:self.batchSize], self.developLabel[0:self.batchSize]
        print(numpy.shape(batchData), numpy.shape(batchLabel))

        predict = self.session.run(fetches=self.parameter['Layer3rd_Pooling'],
                                   feed_dict={self.dataInput: batchData, self.labelInput: batchLabel})
        print(numpy.shape(predict))
