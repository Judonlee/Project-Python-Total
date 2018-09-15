from CTC_Project.Module.LSTM_FinalPooling import LSTM_FinalPooling
from CTC_Project.Loader.IEMOCAP_Spectrogram_Loader import IEMOCAP_Spectrogram_Loader
from time import strftime
import os
import tensorflow

if __name__ == '__main__':
    bands = 30

    for appoint in range(10):
        savepath = 'D:\\GitHubFiles\\FinalPooling\\Bands' + str(bands) + '\\Part' + str(appoint) + '\\'
        os.makedirs(savepath)
        trainData, trainLabel, trainSeq, testData, testLabel, testSeq = IEMOCAP_Spectrogram_Loader(bands=bands,
                                                                                                   appoint=appoint)

        graph = tensorflow.Graph()
        with graph.as_default():
            classifier = LSTM_FinalPooling(trainData=trainData, trainLabel=trainLabel, trainSeqLength=trainSeq,
                                           featureShape=30, numClass=4, learningRate=1e-4, batchSize=128,
                                           hiddenNodules=128, rnnLayers=2)
            print(classifier.information)

            for episode in range(100):
                name = str(episode)
                while len(name) < 4:
                    name = '0' + name

                totalLoss = classifier.Train()
                string = '\rEpisode : ' + str(episode) + '\tTotal Loss : ' + str(totalLoss) + '\t' \
                         + strftime("%Y/%m/%d %H:%M:%S")
                print(string)
                matrix = classifier.Test(testData=testData, testLabel=testLabel, testSequence=testSeq,
                                         savename=savepath + name + '.csv')
                print('Test Part :\n', matrix)

                file = open(savepath + name + '-Matrix.csv', 'w')
                for indexX in range(len(matrix)):
                    for indexY in range(len(matrix[indexX])):
                        file.write(str(matrix[indexX][indexY]) + ',')
                    file.write('\n')
                file.close()
                if episode + 1 == 100:
                    classifier.Save(savepath=savepath + 'NeuralNetwork')
