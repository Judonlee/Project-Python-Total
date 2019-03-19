from DepressionRecognition.Loader import Loader_AutoEncoder
from DepressionRecognition.Model.AutoEncoder import AutoEncoder
from DepressionRecognition.AttentionMechanism.StandardAttention import StandardAttentionInitializer
from DepressionRecognition.AttentionMechanism.LocalAttention import LocalAttentionInitializer
from DepressionRecognition.AttentionMechanism.MonotonicAttention import MonotonicAttentionInitializer, \
    MonotonicChunkwiseAttentionInitializer
import numpy
import os

if __name__ == '__main__':
    attention = StandardAttentionInitializer
    attentionName = 'SA'
    attentionScope = 0

    data, seq = Loader_AutoEncoder()
    print(numpy.shape(data), numpy.shape(seq))
    savepath = 'E:/ProjectData_Depression/Experiment/AutoEncoder/%s-%d/' % (attentionName, attentionScope)
    os.makedirs(savepath)

    classifier = AutoEncoder(data=data, seq=seq, attention=attention, attentionName=attentionName,
                             attentionScope=attentionScope, batchSize=64, learningRate=1E-4)
    for episode in range(20):
        print('\nEpisode %d/100 Total Loss = %f' % (episode, classifier.Train(savepath + '%04d.csv' % episode)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
