import numpy

if __name__ == '__main__':
    for part in ['Decode', 'Logits', 'SoftMax']:
        loadpath = 'E:/CTC_Target_FAU/ARGUMENT/Result-CTC-Origin-WN/Bands-40/%s/' % part

        WAList, UAList = [], []
        for episode in range(80):
            filename = '%04d.csv' % episode
            data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
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
        # print(numpy.argmax(UAList),numpy.argmax(WAList))
