import tensorflow
from __Base.BaseClass import NeuralNetwork_Base
from CASIA_Project.Utils.RegressionEvaluation import MAE_Calculation, RMSE_Calculation
import numpy


class BLSTM(NeuralNetwork_Base):
    def __init__(self, trainData, trainLabel, featureShape, learningRate=1e-4, units=256, layernumber=2, batchSize=8,
                 startFlag=True, graphRevealFlag=True, graphPath='logs/'):
        self.featureShape = featureShape
        self.units = units
        self.layernumber = layernumber
        super(BLSTM, self).__init__(trainData=trainData, trainLabel=trainLabel,
                                    learningRate=learningRate, startFlag=startFlag,
                                    graphRevealFlag=graphRevealFlag, graphPath=graphPath, batchSize=batchSize)
        self.information = '使用BLSTM对于数据进行回归处理'
        for sample in self.parameters.keys():
            self.information += '\n' + str(sample) + '\t' + str(self.parameters[sample])

    def BuildNetwork(self, learningRate):
        self.dataInput = tensorflow.placeholder(dtype=tensorflow.float32, shape=[None, self.featureShape],
                                                name='dataInput')
        self.labelInput = tensorflow.placeholder(dtype=tensorflow.int32, shape=None, name='labelInput')

        ###########################################################################################
        # Layer 1st
        ###########################################################################################

        self.parameters['Layer1st_FC'] = tensorflow.layers.dense(inputs=self.dataInput, units=self.units,
                                                                 activation=tensorflow.nn.tanh, name='Layer1st_FC')

        self.parameters['Layer1st_Reshape'] = tensorflow.reshape(tensor=self.parameters['Layer1st_FC'],
                                                                 shape=[-1, 1, self.units],
                                                                 name='Layer1st_Reshape')

        ###########################################################################################
        # Layer 2nd RNN
        ###########################################################################################
        self.rnnCell = []
        for layers in range(self.layernumber):
            self.parameters['RNN_Cell_Layer' + str(layers)] = tensorflow.contrib.rnn.LSTMCell(self.units)
            self.rnnCell.append(self.parameters['RNN_Cell_Layer' + str(layers)])
        self.parameters['Stack'] = tensorflow.contrib.rnn.MultiRNNCell(self.rnnCell)
        self.parameters['RNN_Outputs'], _ = tensorflow.nn.dynamic_rnn(cell=self.parameters['Stack'],
                                                                      inputs=self.parameters['Layer1st_Reshape'],
                                                                      dtype=tensorflow.float32)
        self.parameters['RNN_Output_Reshape'] = tensorflow.reshape(tensor=self.parameters['RNN_Outputs'][-1],
                                                                   shape=[-1, self.units], name='RNN_Output_Reshape')

        ###########################################################################################
        # Layer 3rd FullConnected
        ###########################################################################################

        self.parameters['Layer3rd_FC'] = tensorflow.layers.dense(inputs=self.parameters['RNN_Output_Reshape'],
                                                                 units=1024, activation=tensorflow.nn.tanh)

        ###########################################################################################
        # Predict&Cost&Train
        ###########################################################################################

        self.predict = tensorflow.layers.dense(inputs=self.parameters['Layer3rd_FC'], units=1,
                                               activation=None, name='Predict')
        self.cost = tensorflow.losses.mean_squared_error(labels=self.labelInput, predictions=self.predict)
        self.train = tensorflow.train.RMSPropOptimizer(learning_rate=learningRate).minimize(self.cost)

    def Train(self):
        totalCost = 0
        for index in range(len(self.data)):
            loss, _ = self.session.run(fetches=[self.cost, self.train],
                                       feed_dict={self.dataInput: self.data[index], self.labelInput: self.label[index]})
            batchOutput = '\rTrain Batch %.4d/%.4d Loss : %f' % (index, len(self.data), loss)
            print(batchOutput, end='')
            totalCost += loss
        return totalCost

    def Test(self, testData, testLabel):
        totalPredict = []
        for index in range(len(testData)):
            predict = self.session.run(fetches=self.predict, feed_dict={self.dataInput: self.data[index]})
            totalPredict.append(predict[0][0])
        return MAE_Calculation(predict=numpy.array(totalPredict) * 63, label=numpy.array(testLabel) * 63), \
               RMSE_Calculation(predict=numpy.array(totalPredict) * 63, label=numpy.array(testLabel) * 63)
