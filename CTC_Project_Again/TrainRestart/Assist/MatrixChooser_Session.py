import os
import numpy

if __name__ == '__main__':
    bans = 30
    for session in range(1, 6):
        UATrace, WATrace = [], []

        loadpathFemale = 'D:/ProjectData/BrandNewCTC/Result-Results-CTC-Right/Bands-%d-Session-%d-%s/SoftMax/' % (
            bans, session, 'Female')
        loadpathMale = 'D:/ProjectData/BrandNewCTC/Result-Results-CTC-Right/Bands-%d-Session-%d-%s/SoftMax/' % (
            bans, session, 'Male')
        for filename in os.listdir(loadpathFemale):
            dataFemale = numpy.genfromtxt(loadpathFemale + filename, dtype=int, delimiter=',')
            dataMale = numpy.genfromtxt(loadpathMale + filename, dtype=int, delimiter=',')
            data = dataFemale + dataMale

            WA, UA = 0, 0
            for index in range(len(data)):
                WA += data[index][index]
                UA += data[index][index] / sum(data[index])
            WA = WA / sum(sum(data))
            UA = UA / len(data)
            # print(WA, UA, sum(sum(data)))
            UATrace.append(UA)
            WATrace.append(WA)

        print(max(UATrace), max(WATrace))
