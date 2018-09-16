import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = r'D:\ProjectData\Current2.txt'
    file = open(loadpath, 'r')
    data = file.readlines()
    file.close()

    treatData = []
    for sample in data:
        if sample[0:7] == 'Episode' or len(sample) < 5: continue

        currentData = []
        for subsample in sample.split(','):
            currentData.append(float(subsample))
        treatData.append(currentData)

    treatData = numpy.array(treatData)
    print(numpy.min(treatData, axis=0))
    plt.title('MAE')
    plt.plot(treatData[:, 0], label='Train')
    plt.plot(treatData[:, 2], label='Develop')
    plt.plot(treatData[:, 4], label='Test')
    plt.legend()
    plt.show()
