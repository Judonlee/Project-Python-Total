import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/Project-CTC-Data/Records-Result-CTC-CRF-Reuse-Restart-CRF/Bands-30-0/'
    UATrace, WATrace = [], []

    maxUAmatrix, maxWAmatrix = [], []
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
        WA, UA = 0, 0
        for index in range(len(data)):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
        WA = WA / sum(sum(data))
        UA = UA / len(data)
        # print(WA, UA, sum(sum(data)))
        UATrace.append(UA)
        WATrace.append(WA)

        if WA == max(WATrace):
            maxWAmatrix = data.copy()
        if UA == max(UATrace):
            maxUAmatrix = data.copy()
    plt.plot(UATrace, label='UA')
    plt.plot(WATrace, label='WA')
    plt.legend()
    plt.show()
    print(max(UATrace), max(WATrace))
