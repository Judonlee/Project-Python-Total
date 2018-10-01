import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\Records-BLSTM-CTC-Normalized\\Result-Decode\\30-9\\'
    WATrace, UATrace = [], []
    for filename in os.listdir(loadpath):
        matrix = numpy.genfromtxt(loadpath + filename, dtype=int, delimiter=',')

        WA = (matrix[0][0] + matrix[1][1] + matrix[2][2] + matrix[3][3]) / sum(sum(matrix))
        UA = 0
        for index in range(len(matrix)):
            UA += matrix[index][index] / sum(matrix[index]) / 4
        print(filename, WA, UA)

        WATrace.append(WA)
        UATrace.append(UA)
    plt.plot(WATrace, label='WA')
    plt.plot(UATrace, label='UA')
    plt.legend()
    # plt.show()
    print(WATrace[numpy.argmax(numpy.array(UATrace))], max(UATrace), numpy.argmax(numpy.array(UATrace)))
