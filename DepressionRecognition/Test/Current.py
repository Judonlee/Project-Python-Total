import numpy
import matplotlib.pylab as plt
import os
import librosa
import random

if __name__ == '__main__':
    plt.subplot(221)
    filename = r'D:\ProjectData\AVEC2017\Step2_VoiceSeparate\train\303_P\Speech_0108.wav'
    data, sr = librosa.load(filename, sr=16000)
    data = numpy.concatenate([data, numpy.zeros([16000 * 4 - len(data)])])
    plt.plot(numpy.arange(0, len(data)) / 16000, data)
    plt.xlabel('Time (Seconds)')
    plt.ylabel('waveform')
    plt.title('Audio')

    plt.subplot(222)
    data = numpy.genfromtxt('Result-Whole.csv', dtype=float, delimiter=',')
    plt.plot(data)
    plt.title('Clinical Interview Attention')
    plt.xlabel('Sentence Number')
    plt.ylabel('Attention Weight')

    plt.subplot(223)
    data = numpy.genfromtxt('Result.csv', dtype=float, delimiter=',')
    plt.plot(numpy.arange(len(data[55])) / 250, data[55])
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Attention Weight')
    plt.title('Speech Recognition')

    plt.subplot(224)
    data = numpy.genfromtxt('Result-DR.csv', dtype=float, delimiter=',')
    plt.plot(numpy.arange(len(data[55])) / 250, data[55])
    plt.xlabel('Time (Seconds)')
    plt.ylabel('Attention Weight')
    plt.title('Depression Recognition')
    # plt.savefig('Visualization.png', format='png', dpi=1000)
    plt.show()
