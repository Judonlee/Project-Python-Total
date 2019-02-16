import tensorflow
from Stock.Loader import Load
from Stock.Model.BLSTM import BLSTM

if __name__ == '__main__':
    data = Load(partName='SZ#002415')[-500:]
    classifier = BLSTM(trainData=data, learningRate=1E-4, considerScope=15)
    for episode in range(1000):
        print('\nEpisode %d/100 Total Loss = %f' % (episode, classifier.Train()))
        classifier.SingleTest(testData=[data[-15:]])
