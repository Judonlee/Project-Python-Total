from python_speech_features import mfcc
import scipy.io.wavfile as wav
import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Voices-Choosed/'
    savepath = 'D:/ProjectData/IEMOCAP/MFCC/'
    for indexA in ['improve']:
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    os.makedirs(os.path.join(savepath, indexA, indexB, indexC, indexD))
                    for indexE in os.listdir(os.path.join(loadpath, indexA, indexB, indexC, indexD)):
                        print(indexA, indexB, indexC, indexD, indexE)
                        fs, audio = wav.read(os.path.join(loadpath, indexA, indexB, indexC, indexD, indexE))
                        result = mfcc(audio)

                        with open(os.path.join(savepath, indexA, indexB, indexC, indexD, indexE + '.csv'), 'w') as file:
                            for indexX in range(len(result)):
                                for indexY in range(len(result[indexX])):
                                    if indexY != 0: file.write(',')
                                    file.write(str(result[indexX][indexY]))
                                file.write('\n')
    # print(numpy.shape(result))
    # print(fs, len(audio))
    # plt.plot(audio)
    # plt.show()
