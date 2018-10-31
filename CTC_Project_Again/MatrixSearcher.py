import os
import numpy
from pprint import pprint

if __name__ == '__main__':
    bands = 60
    WAList, UAList = [], []
    for appoint in range(10):
        loadpath = 'D:/ProjectData/Project-CTC-Data/Records-Result-BLSTM-CTC-CRF-Attention-WA/Bands-%d-%d/' % (
            bands, appoint)
        UATrace, WATrace = [], []

        matrixList = []
        maxUAmatrix, maxWAmatrix = [], []
        for filename in os.listdir(loadpath):
            data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
            # print(data)
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
            matrixList.append(data)
        # print('\n')
        # print('\nAppoint =', appoint)
        print(max(WATrace), max(UATrace))
        # print(numpy.argmax(numpy.array(WATrace)), numpy.argmax(numpy.array(UATrace)))
        WAList.append(numpy.argmax(numpy.array(WATrace)))
        UAList.append(numpy.argmax(numpy.array(UATrace)))
        # pprint(matrixList[numpy.argmax(numpy.array(WATrace))])
        # pprint(matrixList[numpy.argmax(numpy.array(UATrace))])

    print('[', end='')
    for sample in WAList:
        print(sample, end=',')
    print(']')
    print('[', end='')
    for sample in UAList:
        print(sample, end=',')
    print(']')
