import numpy

if __name__ == '__main__':
    for part in ['Decode', 'Logits', 'SoftMax']:
        loadpath = 'E:/CTC_Target_FAU/Result-CTC-Origin/Bands-30/%s/' % part

        WARList, UARList, WAPList, UAPList = [], [], [], []
        for episode in range(100):
            filename = '%04d.csv' % episode
            data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
            # print(data)
            WAR, UAR, WAP, UAP = 0.0, 0.0, 0.0, 0.0
            for index in range(len(data)):
                WAP += data[index][index] / sum(data[index]) * sum(data[index]) / sum(sum(data))
                UAP += data[index][index] / sum(data[index]) / len(data)
                if sum(data[:, index]) != 0:
                    WAR += data[index][index] / sum(data[index]) * sum(data[index]) / sum(sum(data))
                    UAR += data[index][index] / sum(data[index]) / len(data)
            # print(WAR, UAR, WAP, UAP)
            # exit()

            WARList.append(WAR)
            WAPList.append(WAP)
            UARList.append(UAR)
            UAPList.append(UAP)

        print(max(UARList), '\t', max(WARList), '\t', max(UAPList), '\t', max(WAPList))
