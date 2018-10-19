from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader_Npy, IEMOCAP_SeqLabelLoader
import numpy
import os

if __name__ == '__main__':
    # for bands in [30]:
    #     for appoint in range(10):
    #         trainData, trainLabel, trainSeq, trainScription, testData, testLabel, testSeq, testScription = \
    #             IEMOCAP_Loader_Npy(
    #                 loadpath='D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper/Bands-%d-%d/' % (bands, appoint))
    #         for index in range(10):
    #             print(numpy.shape(trainData[index]), trainSeq[index], trainScription[index])
    #         exit()
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU/'
    savepath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU-Npy-Improve/'

    for appoint in range(10):
        os.makedirs(savepath + 'Appoint-' + str(appoint))
        trainScription, testScription = [], []
        for indexA in ['improve']:
            for indexB in os.listdir(os.path.join(loadpath, indexA)):
                for indexC in range(1, 6):
                    for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, 'Session' + str(indexC))):
                        for indexE in os.listdir(
                                os.path.join(loadpath, indexA, indexB, 'Session' + str(indexC), indexD)):
                            print(indexA, indexB, indexC, indexD, indexE)

                            file = open(os.path.join(loadpath, indexA, indexB, 'Session' + str(indexC), indexD, indexE),
                                        'r')
                            data = file.read()
                            file.close()
                            label = numpy.ones(data.count(' ') + 1)
                            if indexD == 'ang': label = label * 0
                            if indexD == 'exc' or indexD == 'hap': label = label * 1
                            if indexD == 'neu': label = label * 2
                            if indexD == 'sad': label = label * 3
                            if ['Female', 'Male'].index(indexB) * 5 + indexC - 1 == appoint:
                                testScription.append(label)
                            else:
                                trainScription.append(label)
        print(numpy.shape(trainScription), numpy.shape(testScription))
        numpy.save(savepath + 'Appoint-' + str(appoint) + '/TrainTranscription.npy', trainScription)
        numpy.save(savepath + 'Appoint-' + str(appoint) + '/TestTranscription.npy', testScription)
        # exit()
