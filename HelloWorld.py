import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_SpeechRecognition/IEMOCAP-Origin-Result/Bands30/Session0-LR/'
    totalData = []
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',') / 2943 * 32
        totalData.append(data)
        print(data)
    plt.plot(totalData)
    plt.xlabel('Train Episode')
    plt.ylabel('Batch Loss')
    plt.title('Training Loss Line')
    plt.show()
