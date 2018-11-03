import numpy
import os

if __name__ == '__main__':
    matrixWA, matrixUA = numpy.zeros((4, 4)), numpy.zeros((4, 4))
    for appoint in range(10):
        loadpath = 'D:/ProjectData/Records-OpenSmile/eGeMAPSv01a-Appoint-%d/Logits/' % appoint

        matrixList = []
        uaList, waList = [], []
        for filename in os.listdir(loadpath):
            if os.path.isdir(loadpath + filename): continue
            data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
            # print(data)
            WA, UA = 0, 0
            for index in range(len(data)):
                WA += data[index][index]
                UA += data[index][index] / sum(data[index])
            WA = WA / sum(sum(data))
            UA = UA / len(data)

            waList.append(WA)
            uaList.append(UA)
            matrixList.append(data)
        matrixWA += matrixList[numpy.argmax(numpy.array(waList))]
        matrixUA += matrixList[numpy.argmax(numpy.array(uaList))]
    for indexX in range(4):
        for indexY in range(4):
            if indexY != 0: print('\t', end='')
            print(matrixWA[indexX][indexY], end='')
        print()
    print('\n\n')
    for indexX in range(4):
        for indexY in range(4):
            if indexY != 0: print('\t', end='')
            print(matrixUA[indexX][indexY], end='')
        print()
