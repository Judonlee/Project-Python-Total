import os
import librosa
from scipy import signal
import numpy

if __name__ == '__main__':
    m_bands = 120
    loadpath = 'D:\\ProjectData\\FAU-AEC-Treated\\IS2009-Class5\\'
    savepath = 'D:\\ProjectData\\FAU-AEC-Treated\\IS2009-Class5-Csv\\Bands-%d\\' % m_bands

    s_rate = 16000
    win_length = int(0.025 * s_rate)  # Window length 15ms, 25ms, 50ms, 100ms, 200ms
    hop_length = int(0.010 * s_rate)  # Window shift  10ms
    n_fft = win_length

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            os.makedirs(savepath + indexA + '\\' + indexB)
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                print(indexA, indexB, indexC)
                y, sr = librosa.load(loadpath + indexA + '\\' + indexB + '\\' + indexC, sr=16000)

                D = numpy.abs(librosa.stft(y, n_fft=n_fft, win_length=win_length, hop_length=hop_length,
                                           window=signal.hamming, center=False)) ** 2
                S = librosa.feature.melspectrogram(S=D, n_mels=m_bands)
                gram = librosa.power_to_db(S, ref=numpy.max)
                gram = numpy.transpose(gram, (1, 0))
                # print(numpy.shape(gram))

                file = open(savepath + indexA + '\\' + indexB + '\\' + indexC + '.csv', 'w')
                for indexX in range(len(gram)):
                    for indexY in range(len(gram[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(gram[indexX][indexY]))
                    file.write('\n')
                file.close()
