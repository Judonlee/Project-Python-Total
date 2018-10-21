import librosa
import os
import numpy


def VoiceLoader(loadpath, appoint=0):
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            print(indexA, indexB)
            for indexC in ['ang', 'exc', 'hap', 'neu', 'sad']:
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    wav, sr = librosa.load(path=os.path.join(loadpath, indexA, indexB, indexC, indexD), sr=16000)
                    currentSeq = int(len(wav) / sr * 25)
                    if indexC == 'ang': currentLabel = [1, 0, 0, 0]
                    if indexC == 'exc' or indexC == 'hap': currentLabel = [0, 1, 0, 0]
                    if indexC == 'neu': currentLabel = [0, 0, 1, 0]
                    if indexC == 'sad': currentLabel = [0, 0, 0, 1]
                    if ['Female', 'Male'].index(indexA) * 5 + ['Session1', 'Session2', 'Session3', 'Session4',
                                                               'Session5'].index(indexB) == appoint:
                        testData.append(wav)
                        testLabel.append(currentLabel)
                        testSeq.append(currentSeq)
                    else:
                        trainData.append(wav)
                        trainLabel.append(currentLabel)
                        trainSeq.append(currentSeq)
    print(numpy.sum(trainLabel, axis=0), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


if __name__ == '__main__':
    VoiceLoader(loadpath='D:/ProjectData/IEMOCAP/IEMOCAP-Voices/improve/')
