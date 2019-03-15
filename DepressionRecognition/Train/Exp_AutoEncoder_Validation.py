from DepressionRecognition.Loader import Loader_AutoEncoder
from DepressionRecognition.Model.AutoEncoder import AutoEncoder
import numpy
import os

if __name__ == '__main__':
    data, seq = Loader_AutoEncoder()
    print(numpy.shape(data), numpy.shape(seq))
    savepath = 'E:/ProjectData_Depression/Experiment/AutoEncoder/WithoutAttention/'
    os.makedirs(savepath)

    classifier = AutoEncoder(data=data, seq=seq)
    for episode in range(100):
        print('\nEpisode %d/100 Total Loss = %f' % (episode, classifier.Train()))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
