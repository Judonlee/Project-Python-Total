import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/CTC_Target/Result-CTC-Origin/'
    bands = 30
    usedPart = 'Logits'
    for session in range(1, 6):
        FemaleUAList, FemaleWAList, MaleUAList, MaleWAList = [], [], [], []
        for episode in range(100):
            appoint = 'Bands-%d-Session-%d-%s/%s/' % (bands, session, 'Female', usedPart)
            filename = '%04d.csv' % episode
            data = numpy.genfromtxt(fname=loadpath + appoint + filename, dtype=float, delimiter=',')
            WA, UA = 0, 0
            for index in range(len(data)):
                WA += data[index][index]
                UA += data[index][index] / sum(data[index])
            WA = WA / sum(sum(data))
            UA = UA / len(data)
            FemaleUAList.append(UA)
            FemaleWAList.append(WA)

            appoint = 'Bands-%d-Session-%d-%s/%s/' % (bands, session, 'Male', usedPart)
            filename = '%04d.csv' % episode
            data = numpy.genfromtxt(fname=loadpath + appoint + filename, dtype=float, delimiter=',')
            WA, UA = 0, 0
            for index in range(len(data)):
                WA += data[index][index]
                UA += data[index][index] / sum(data[index])
            WA = WA / sum(sum(data))
            UA = UA / len(data)
            MaleUAList.append(UA)
            MaleWAList.append(WA)

        # print(max(FemaleWAList), max(FemaleUAList), max(MaleWAList), max(MaleUAList))
        print(FemaleWAList[numpy.argmax(MaleWAList)], FemaleUAList[numpy.argmax(MaleWAList)],
              MaleWAList[numpy.argmax(FemaleUAList)], MaleUAList[numpy.argmax(FemaleWAList)])
