import os
import librosa
from scipy import signal
import numpy


def Extraction(loadpath, savepath, bands):
    s_rate = 16000
    win_length = int(0.025 * s_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
    hop_length = int(0.010 * s_rate)  # Window shift  10ms
    n_fft = win_length

    y, sr = librosa.load(loadpath, sr=16000)

    try:
        D = numpy.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length, window=signal.hamming,
                                   center=False)) ** 2
        S = librosa.feature.melspectrogram(S=D, n_mels=bands)
        gram = librosa.power_to_db(S, ref=numpy.max)
        gram = numpy.transpose(gram, (1, 0))
        print(numpy.shape(gram))

        numpy.save(savepath, gram)

        # file = open(savepath, 'w')
        # for indexX in range(len(gram)):
        #     for indexY in range(len(gram[indexX])):
        #         if indexY != 0: file.write(',')
        #         file.write(str(gram[indexX][indexY]))
        #     file.write('\n')
        # file.close()
    except:
        pass


if __name__ == '__main__':
    bands = 40
    loadpath = 'D:/ProjectData/AVEC2017-Separate/'
    savepath = 'D:/ProjectData/AVEC2017-Bands%d/Step1_Npy/' % bands

    for indexA in os.listdir(loadpath):
        if os.path.isfile(os.path.join(loadpath, indexA)): continue
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            if os.path.exists(os.path.join(savepath, indexA, indexB)): continue
            os.makedirs(os.path.join(savepath, indexA, indexB))
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                if indexC[-3:] != 'wav': continue
                if indexC.find('Participant') == -1: continue
                print(indexA, indexB, indexC)
                if os.path.exists(os.path.join(savepath, indexA, indexB, indexC + '.npy')): continue
                Extraction(loadpath=os.path.join(loadpath, indexA, indexB, indexC),
                           savepath=os.path.join(savepath, indexA, indexB, indexC + '.npy'), bands=bands)
