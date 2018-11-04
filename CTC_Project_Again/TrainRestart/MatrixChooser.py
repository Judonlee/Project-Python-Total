import os
import numpy

if __name__ == '__main__':
    bans = 30
    part = 'Logits'
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            if session > 2:
                print()
                continue
            UATrace, WATrace = [], []
            loadpath = 'D:/ProjectData/BrandNewCTC/Data-Result-Changed-Left-TripleBLSTM/Bands-%d-Session-%d-%s/%s/' % (
                bans, session, gender, part)
            if not os.path.exists(loadpath): continue
            for file in os.listdir(loadpath):
                if file[-3:] != 'csv': continue
                data = numpy.genfromtxt(loadpath + file, dtype=int, delimiter=',')
                WA, UA = 0, 0
                for index in range(len(data)):
                    WA += data[index][index]
                    UA += data[index][index] / sum(data[index])
                WA = WA / sum(sum(data))
                UA = UA / len(data)
                # print(WA, UA, sum(sum(data)))
                UATrace.append(UA)
                WATrace.append(WA)

            loadpath = 'D:/ProjectData/BrandNewCTC/Data-Result-Changed-Left-TripleBLSTM/Bands-%d-Session-%d-%s/%s/' % (
                bans, session, gender, part)
            if not os.path.exists(loadpath): continue
            for file in os.listdir(loadpath):
                if file[-3:] != 'csv': continue
                data = numpy.genfromtxt(loadpath + file, dtype=int, delimiter=',')
                WA, UA = 0, 0
                for index in range(len(data)):
                    WA += data[index][index]
                    UA += data[index][index] / sum(data[index])
                WA = WA / sum(sum(data))
                UA = UA / len(data)
                # print(WA, UA, sum(sum(data)))
                UATrace.append(UA)
                WATrace.append(WA)
            print(max(UATrace), '\t', max(WATrace))
