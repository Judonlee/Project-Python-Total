import os


def TrainTimes200(classifier, savepath):
    if not os.path.exists(savepath): os.makedirs(savepath)
    for episode in range(100):
        print('\nEpisode %d : Total Loss = %f' % (episode, classifier.Train(learningRate=1E-3)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)

    for episode in range(100, 200):
        print('\nEpisode %d : Total Loss = %f' % (episode, classifier.Train(learningRate=1E-4)))
        classifier.Save(savepath=savepath + '%04d-Network' % episode)
