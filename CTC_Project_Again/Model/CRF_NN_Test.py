import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
import tensorflow.contrib.crf as crf
import numpy
from __Base.Shuffle import Shuffle


class CRF_Test(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, trainSeqLength, featureShape, numClass, hiddenNodules=128, batchSize=32,
                 learningRate=1e-4, startFlag=True, graphRevealFlag=True, graphPath='logs/', occupyRate=-1):
        self.featureShape = featureShape
        self.seqLen = trainSeqLength
        self.numClass = numClass
        self.hiddenNodules = hiddenNodules
        super(CRF_Test, self).__init__(trainData=trainData, trainLabel=trainLabel, batchSize=batchSize,
                                       learningRate=learningRate, startFlag=startFlag, graphRevealFlag=graphRevealFlag,
                                       graphPath=graphPath, occupyRate=occupyRate)

        self.information = 'This Model uses the Final_Pooling to testify the validation of the model.'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None, None], name='labelInput')
        self.seqLenInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=[None], name='seqLenInput')

        self.parameters['BatchSize'] = tensorflow.shape(input=self.dataInput, name='BatchSize')[0]
        self.parameters['TimeStep'] = tensorflow.shape(input=self.dataInput, name='TimeStep')[1]

        ###################################################################################################
        # Network Start
        ###################################################################################################

        self.parameters['Data_Reshape'] = tensorflow.reshape(tensor=self.dataInput, shape=[-1, self.featureShape],
                                                             name='Data_Reshape')
        self.parameters['Layer1st_FC'] = tensorflow.layers.dense(inputs=self.parameters['Data_Reshape'],
                                                                 units=self.numClass, activation=None,
                                                                 name='Layer1st_FC')
        self.parameters['Logits'] = tensorflow.reshape(tensor=self.parameters['Layer1st_FC'],
                                                       shape=[self.parameters['BatchSize'], self.parameters['TimeStep'],
                                                              self.numClass], name='Logits')

        self.parameters['LogLikelihood'], self.parameters['TransitionParams'] = crf.crf_log_likelihood(
            inputs=self.parameters['Logits'], tag_indices=self.labelInput, sequence_lengths=self.seqLenInput)
        self.parameters['Loss'] = tensorflow.reduce_mean(input_tensor=-self.parameters['LogLikelihood'])
        self.train = tensorflow.train.AdamOptimizer(learning_rate=learningRate).minimize(self.parameters['Loss'])

    def Train(self):
        trainData, trainLabel, trainSeq = Shuffle(data=self.data, label=self.label, seqLen=self.seqLen)

        startPosition = 0
        totalLoss = 0
        while startPosition < len(trainData):
            batchData, batchLabel = [], []
            batachSeq = trainSeq[startPosition:startPosition + self.batchSize]

            maxLen = max(trainSeq[startPosition:startPosition + self.batchSize])
            for index in range(startPosition, min(startPosition + self.batchSize, len(trainData))):
                currentData = numpy.concatenate(
                    (trainData[index], numpy.zeros((maxLen - len(trainData[index]), len(trainData[index][0])))), axis=0)
                batchData.append(currentData)

                currentLabel = numpy.concatenate((trainLabel[index], numpy.zeros(maxLen - len(trainLabel[index]))),
                                                 axis=0)
                batchLabel.append(currentLabel)

            loss, _ = self.session.run(fetches=[self.parameters['Loss'], self.train],
                                       feed_dict={self.dataInput: batchData, self.labelInput: batchLabel,
                                                  self.seqLenInput: batachSeq})
            totalLoss += loss
            output = '\rBatch : %d/%d \t Loss : %f' % (startPosition, len(trainData), loss)
            print(output, end='')
            startPosition += self.batchSize
        return totalLoss
