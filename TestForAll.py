import os
import numpy

if __name__ == '__main__':
    bands, appoint = 60, 0
    loadpathA = 'D:/ProjectData/Records-Result-BLSTM-CTC-CRF-Improve-UA/Bands-%d-%d/' % (bands, appoint)
    loadpathB = 'D:/ProjectData/Unknown/Records-Result-BLSTM-CTC-CRF-Improve-UA/Bands-%d-%d' % (bands, appoint)
    for filename in os.listdir(loadpathA):
        print(filename)
        dataX = numpy.genfromtxt(os.path.join(loadpathA, filename), dtype=int, delimiter=',')
        dataY = numpy.genfromtxt(os.path.join(loadpathB, filename), dtype=int, delimiter=',')
        for indexX in range(4):
            for indexY in range(4):
                if dataX[indexX][indexY] != dataY[indexX][indexY]: print('ERROR')
