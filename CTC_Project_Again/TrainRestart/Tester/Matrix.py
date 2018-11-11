import numpy
import os

if __name__ == '__main__':
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            loadpath = 'D:/ProjectData/CTC_Changed/Result-Result-BLSTM-CTC-CRF-Single-Concat/Bands-30-Session-%d-%s/' % (
                session, gender)
            if not os.path.exists(loadpath):
                print()
                continue
            waList, uaList = [], []
            matrix = []
            for filename in os.listdir(loadpath):
                data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
                WA, UA = 0, 0
                for index in range(len(data)):
                    WA += data[index][index]
                    UA += data[index][index] / sum(data[index])
                WA = WA / sum(sum(data))
                UA = UA / len(data)
                waList.append(WA)
                uaList.append(UA)

                if UA == max(uaList):
                    matrix = data.copy()
            print(max(waList), max(uaList))
            # print(matrix)
