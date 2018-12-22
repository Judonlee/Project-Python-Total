import numpy
import os

if __name__ == '__main__':
    for gender in ['F', 'M']:
        for session in range(1, 7):
            loadpath = 'E:/CTC_Target_MSP/Result-CTC-Origin-MSP/Bands-30/Session-%d-%s/SoftMax/' % (session, gender)

            if not os.path.exists(loadpath):
                print()
                continue
            WAList, UAList = [], []
            for episode in range(100):
                filename = '%04d.csv' % episode
                data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
                WA, UA = 0, 0
                for index in range(len(data)):
                    WA += data[index][index]
                    UA += data[index][index] / sum(data[index])
                WA = WA / sum(sum(data))
                UA = UA / len(data)

                WAList.append(WA)
                UAList.append(UA)
            print(max(WAList), '\t', max(UAList))
