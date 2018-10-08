import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\Project-CTC-Data\\Records-Result-CTC-LR1e-3-RMSP\\Bands-60-3\\SoftMax\\'
    UATrace, WATrace = [], []
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
        WA, UA = 0, 0
        for index in range(len(data)):
            WA += data[index][index]
            UA += data[index][index] / sum(data[index])
        WA = WA / sum(sum(data))
        UA = UA / len(data)
        print(WA, UA, sum(sum(data)))
        UATrace.append(UA)
        WATrace.append(WA)
    print('\n')
    print(max(WATrace), max(UATrace))
