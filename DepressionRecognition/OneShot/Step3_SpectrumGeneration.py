import os
import librosa
from scipy import signal
import numpy
import shutil


def Extraction(loadpath, savepath, bands):
    secondRate = 16000
    winLen = int(0.025 * secondRate)
    hopLen = int(0.010 * secondRate)
    numberFFT = winLen

    y, sr = librosa.load(loadpath, sr=secondRate)

    try:
        D = numpy.abs(librosa.stft(y, n_fft=numberFFT, win_length=winLen, hop_length=hopLen, window=signal.hamming,
                                   center=False)) ** 2
        S = librosa.feature.melspectrogram(S=D, n_mels=bands)
        gram = librosa.power_to_db(S, ref=numpy.max)
        gram = numpy.transpose(gram, (1, 0))

        file = open(savepath, 'w')
        for indexX in range(len(gram)):
            for indexY in range(len(gram[indexX])):
                if indexY != 0: file.write(',')
                file.write(str(gram[indexX][indexY]))
            file.write('\n')
        file.close()
    except:
        pass


if __name__ == '__main__':
    bands = 40
    loadpath = 'D:/ProjectData/AVEC2017/Step2_VoiceSeparate/'
    savepath = 'D:/ProjectData/AVEC2017/Step3_SpectrumFeatures/'
    for foldA in os.listdir(loadpath)[2:3]:
        for foldB in os.listdir(os.path.join(loadpath, foldA))[100:200]:
            print(foldA, foldB)
            if os.path.exists(os.path.join(savepath, foldA, foldB)): continue
            os.makedirs(os.path.join(savepath, foldA, foldB))
            for filename in os.listdir(os.path.join(loadpath, foldA, foldB)):
                if filename[-3:] == 'csv':
                    shutil.copy(src=os.path.join(loadpath, foldA, foldB, filename),
                                dst=os.path.join(savepath, foldA, foldB, filename))
                    continue
                Extraction(loadpath=os.path.join(loadpath, foldA, foldB, filename),
                           savepath=os.path.join(savepath, foldA, foldB, filename[0:filename.find('.')] + '.csv'),
                           bands=bands)
