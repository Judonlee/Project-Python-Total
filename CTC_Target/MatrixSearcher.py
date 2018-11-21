import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/CTC_Target/Result-CTC-Origin/'
    bands = 40
    usedPart = 'Logits'
    for gender in ['Female', 'Male']:
        for session in range(1, 6):
            appoint = 'Bands-%d-Session-%d-%s/%s/' % (bands, session, gender, usedPart)
            if not os.path.exists(loadpath + appoint):
                print()
                continue

            WAList, UAList = [], []
            for episode in range(100):
                filename = '%04d.csv' % episode
                data = numpy.genfromtxt(fname=loadpath + appoint + filename, dtype=float, delimiter=',')
                WA, UA = 0, 0
                for index in range(len(data)):
                    WA += data[index][index]
                    UA += data[index][index] / sum(data[index])
                WA = WA / sum(sum(data))
                UA = UA / len(data)

                # print(WA, UA)
                # exit()
                WAList.append(WA)
                UAList.append(UA)
            print(max(WAList), '\t', max(UAList))
