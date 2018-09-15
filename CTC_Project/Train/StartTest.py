from CTC_Project.Module.CTC_Reconfiguration import CTC
from CTC_Project.Loader.IEMOCAP_Transaction_Loader import CTC_Loader
from CTC_Project.Loader.IEMOCAP_CTC_Loader import IEMOCAP_CTC_Loader
from time import strftime
import os

if __name__ == '__main__':
    savepath = 'D:\\ProjectData\\Results-Labels\\Bands30-Again\\'
    os.makedirs(savepath)
    train_inputs, train_targets, train_seq_len = IEMOCAP_CTC_Loader(
        datafold='D:\\ProjectData\\IEMOCAP-Normalized\\Bands30\\',
        labelfold='D:\\ProjectData\\IEMOCAP-Label-Words\\')
    print(len(train_inputs))
    # exit()
    classifier = CTC(trainData=train_inputs, trainLabel=train_targets, trainSeqLength=train_seq_len,
                     featureShape=30, numClass=6, learningRate=0.5e-5, rnnLayers=1, hiddenNodules=1024)
    print(classifier.information)
    for episode in range(1000):
        name = str(episode)
        while len(name) < 5:
            name = '0' + name

        print('Episode', episode, ':', classifier.Train(), strftime("%Y/%m/%d %H:%M:%S"))
        classifier.PredictOutput(testData=train_inputs, testSequenceLength=train_seq_len,
                                 filename=savepath + name + '.txt')
        if episode != 0 and episode % 10 == 0:
            classifier.Save(savepath + name)
