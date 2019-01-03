import numpy
import os

THRESHOLD = 32

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step5-NonNodules-MediaPosition/'
    savepath = 'E:/LIDC/TreatmentTrace/Step6-NonNodules-Assembly/'
    os.makedirs(savepath)
    totalCounter = 0
    for filename in os.listdir(loadpath):
        print(filename)
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',')
        if len(numpy.shape(data)) == 1: continue
        data = numpy.reshape(data, [-1, 3])

        totalRecords = []
        totalRecords.append(numpy.concatenate((data[0], [1])))
        for indexA in range(1, numpy.shape(data)[0]):
            flag = False
            for sample in totalRecords:
                distance = 0
                for indexB in range(numpy.shape(data)[1]):
                    distance += abs(data[indexA][indexB] - sample[indexB])
                if distance < THRESHOLD:
                    sample[-1] += 1
                    flag = True
                    break
            if not flag: totalRecords.append(numpy.concatenate((data[indexA], [1])))
        print(totalRecords)

        with open(os.path.join(savepath, filename), 'w') as file:
            for indexX in range(len(totalRecords)):
                for indexY in range(len(totalRecords[indexX])):
                    if indexY != 0: file.write(',')
                    file.write(str(totalRecords[indexX][indexY]))
                file.write('\n')
