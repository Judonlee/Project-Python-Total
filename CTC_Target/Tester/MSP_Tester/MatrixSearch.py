import numpy
import os

if __name__ == '__main__':
    counter = 0
    for gender in ['F', 'M']:
        for session in range(1, 7):
            loadpath = 'E:/CTC_Target_MSP/DNN-DropOut/GeMAPSv01a-Result/Session%d-%s/' % (session, gender)
            # print(loadpath)
            if not os.path.exists(loadpath):
                print()
                continue
            WAList, UAList = [], []
            for episode in range(100):
                filename = '%04d.csv' % episode
                if not os.path.exists(loadpath + filename): break
                data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
                counter += numpy.sum(data)
                WA, UA = 0, 0
                for index in range(len(data)):
                    WA += data[index][index]
                    UA += data[index][index] / sum(data[index])
                WA = WA / sum(sum(data))
                UA = UA / len(data)

                WAList.append(WA)
                UAList.append(UA)
            print(max(WAList), '\t', max(UAList))
    print(counter)