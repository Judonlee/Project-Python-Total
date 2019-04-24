import numpy

if __name__ == '__main__':
    data = numpy.genfromtxt(fname='InputRecord.csv', dtype=str, delimiter=',')
    sequence = numpy.genfromtxt(fname='Sequence.csv', dtype=str, delimiter=',')

    with open('Damage.csv', 'w') as file:
        for name in sequence:
            file.write(name + ',')
            counter = 0.0
            times = 0
            flag = False

            for index in range(len(data)):
                findFlag = False
                if data[index][0] == name:
                    file.write(data[index][1])
                    counter += float(data[index][1])
                    findFlag = True
                    times += 1

                if findFlag and int(data[index][1]) < 0:
                    file.write('(撞刀惩罚)+')
                    times -= 1
                    continue

                if findFlag and index != len(data) - 1 and (
                        data[index][-1] != data[index + 1][-1] or int(data[index + 1][1]) < 0):
                    file.write('(%s击破)+' % data[index][-1])
                    flag = True
                    times -= 1
                    continue

                if findFlag and flag:
                    flag = False
                    file.write('(%s)' % data[index][-1])
                if findFlag: file.write(',')

            for _ in range(times, 3):
                file.write('0,')
            file.write(str(counter))
            file.write('\n')

    totalLoss = 0
    with open('Score.csv', 'w') as file:
        for name in sequence:
            file.write(name + ',')
            counter = 0.0
            times = 0
            flag = False

            for index in range(len(data)):
                findFlag = False
                if data[index][0] == name:
                    score = int(float(data[index][1]) * float(data[index][2]))

                    file.write(str(score))
                    counter += score
                    findFlag = True
                    times += 1

                if findFlag and int(data[index][1]) < 0:
                    file.write('(撞刀惩罚)+')
                    times -= 1
                    continue

                if findFlag and index != len(data) - 1 and (
                        data[index][-1] != data[index + 1][-1] or int(data[index + 1][1]) < 0):
                    file.write('(%s击破)+' % data[index][-1])
                    flag = True
                    times -= 1
                    continue

                if findFlag and flag:
                    flag = False
                    file.write('(%s)' % data[index][-1])
                if findFlag: file.write(',')

            flag = True
            for _ in range(times, 3):
                file.write('0,')
                if flag:
                    totalLoss += 3 - times
                    print(name, 3 - times)
                    flag = False
            file.write(str(counter))
            file.write('\n')

    # print(data)
    # print(sequence)

    print('共计缺%d刀' % totalLoss)
