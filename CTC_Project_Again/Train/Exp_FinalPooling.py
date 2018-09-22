from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader
from CTC_Project_Again.Model.LSTM_FinalPooling import LSTM_FinalPooling

if __name__ == '__main__':
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
        IEMOCAP_Loader(loadpath='F:\\Project-CTC-Data\\Npy\\Bands30\\', appoint=0)
    classifier = LSTM_FinalPooling(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq, featureShape=30,
                                   numClass=4)
    print(classifier.information)
    for episode in range(100):
        loss = classifier.Train()
        print('\rEpisode %d/%d Total Loss : %f' % (episode, 100, loss))
        print('Train Part :')
        print(classifier.Test(testData=trainData, testLabel=trainLabel, testSeq=trainSeq))
        print('Test Part :')
        print(classifier.Test(testData=testData, testLabel=testLabel, testSeq=testSeq))
        print('\n')
