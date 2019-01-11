import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadname = ['IEMOCAP-Origin-Result', 'IEMOCAP-LA-3-Result', 'IEMOCAP-LA-5-Result', 'IEMOCAP-LA-7-Result',
                'IEMOCAP-COMA-5-Result']
    for name in loadname:
        loadpath = 'E:/ProjectData_SpeechRecognition/%s/Bands30/Session0/' % name
        totalData = []
        for filename in os.listdir(loadpath):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',') / 2943 * 32
            totalData.append(data)
            # print(data)
        plt.plot(totalData, label=name)
        print('%s\t%.04f' % (name, min(totalData)))
    plt.xlabel('Train Episode')
    plt.ylabel('Batch Loss')
    plt.title('Training Loss Line - Bands30')
    plt.legend()
    plt.show()
