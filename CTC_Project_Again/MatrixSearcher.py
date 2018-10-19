import os
import numpy
from pprint import pprint

if __name__ == '__main__':
    bands = 120
    matrixList = []
    WAList, UAList = [], []
    for appoint in range(8):
        loadpath = 'D:/ProjectData/Project-CTC-Data/Records-Result-CRF-BLSTM-Class4-Tanh/Bands-%d-%d/' \
                   % (bands, appoint)
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
        # print('\nAppoint =', appoint)
        print(max(WATrace), max(UATrace))
        # print(numpy.argmax(numpy.array(WATrace)), numpy.argmax(numpy.array(UATrace)))
        WAList.append(numpy.argmax(numpy.array(WATrace)))
        UAList.append(numpy.argmax(numpy.array(UATrace)))
        # pprint(maxWAmatrix)
        # pprint(maxUAmatrix)

    print('[', end='')
    for sample in WAList:
        print(sample, end=',')
    print(']')
    print('[', end='')
    for sample in UAList:
        print(sample, end=',')
    print(']')
