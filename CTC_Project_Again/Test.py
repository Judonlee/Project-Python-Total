import os
import numpy

if __name__ == '__main__':
    for appoint in range(10):
        loadpath = 'D:\\ProjectData\\Project-CTC-Data\\Records-Result-CRF-BLSTM-Class4-Tanh\\Bands-30-%d\\' % appoint
        UATrace, WATrace = [], []
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
        # print('\n')
        print(max(WATrace), max(UATrace))
