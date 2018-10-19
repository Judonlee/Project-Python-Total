from CTC_Project_Again.Loader.IEMOCAP_Loader import IEMOCAP_Loader, IEMOCAP_TranscriptionLoader
import tensorflow
import os
import numpy

if __name__ == '__main__':
    for bands in [30, 40, 60, 80, 100, 120]:
        for appoint in range(10):
            savepath = 'D:/ProjectData/Project-CTC-Data/Npy-TotalWrapper-Improve/Bands-' + str(
                bands) + '-' + str(appoint) + '/'
            if os.path.exists(savepath): continue
            os.makedirs(savepath)
            graph = tensorflow.Graph()
            with graph.as_default():
                trainData, trainLabel, trainSeq, testData, testLabel, testSeq = \
                    IEMOCAP_Loader(loadpath='D:/ProjectData/Project-CTC-Data/Npy-Normalized/Bands' + str(bands) + '/',
                                   appoint=appoint)
                trainScription, testTranscription = IEMOCAP_TranscriptionLoader(
                    loadpath='D:/ProjectData/Project-CTC-Data/Transcription-SingleNumber-Class5/', appoint=appoint)
                numpy.save(savepath + 'TrainData.npy', [trainData, trainLabel, trainSeq, trainScription])
                numpy.save(savepath + 'TestData.npy', [testData, testLabel, testSeq, testTranscription])
