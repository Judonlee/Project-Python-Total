import os
import numpy

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step6-NonNodules-Assembly/'
    totalCounter = 0
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(os.path.join(loadpath, filename), dtype=int, delimiter=',')
        data = numpy.reshape(data, [-1, 4])
        for index in range(len(data)):
            if data[index][3] >= 2: totalCounter += data[index][3]
    print(totalCounter)
