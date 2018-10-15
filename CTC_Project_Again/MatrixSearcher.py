import os
import numpy
from pprint import pprint

if __name__ == '__main__':
    matrixList = []
    for appoint in range(6):
        loadpath = 'D:\\ProjectData\\Project-CTC-Data\\Records-Result-CRF-BLSTM-Class4-Tanh\\Bands-80-%d\\' % appoint
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
        # print('\n')
        print(max(WATrace), max(UATrace))
        #pprint(maxWAmatrix)
        #pprint(maxUAmatrix)
