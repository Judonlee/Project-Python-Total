import numpy
import os

if __name__ == '__main__':
    loadpath = 'E:/CTC_Target_MSP/Result-SVM/eGeMAPSv01a-Result/'
    for cSearch in range(12):
        for gammaSearch in range(-15, -3, 1):
            c = numpy.power(2, cSearch)
            gamma = 1 / numpy.power(2, -gammaSearch)

            if not os.path.exists(loadpath + 'C=%f,Gamma=%f' % (c, gamma)):
                exit()
            WAList, UAList = [], []
            for filename in os.listdir(os.path.join(loadpath, 'C=%f,Gamma=%f' % (c, gamma))):
                # print(filename)

                if len(os.listdir(os.path.join(loadpath, 'C=%f,Gamma=%f' % (c, gamma)))) != 12:
                    exit()

                data = numpy.genfromtxt(fname=loadpath + 'C=%f,Gamma=%f/' % (c, gamma) + filename, dtype=float,
                                        delimiter=',')
                if len(numpy.shape(data)) == 1: exit()
                # print(data)
                WA, UA = 0, 0
                for index in range(len(data)):
                    WA += data[index][index]
                    UA += data[index][index] / sum(data[index])
                WA = WA / sum(sum(data))
                UA = UA / len(data)

                WAList.append(WA)
                UAList.append(UA)
            # print(WAList)
            print(numpy.average(UAList), end='\t')
        print()
