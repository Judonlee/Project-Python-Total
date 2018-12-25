import numpy
import os

if __name__ == '__main__':
    for gender in ['F', 'M']:
        for session in range(1, 7):
            loadpath = 'D:/ProjectData/MSP-IMPROVE/OpenSmile/IS13-Result/'

            filename = 'Session-%d-%s.csv' % (session, gender)

            data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
            # print(data)
            WA, UA = 0, 0
            for index in range(len(data)):
                WA += data[index][index]
                UA += data[index][index] / sum(data[index])
            WA = WA / sum(sum(data))
            UA = UA / len(data)

            print(WA, '\t', UA)
