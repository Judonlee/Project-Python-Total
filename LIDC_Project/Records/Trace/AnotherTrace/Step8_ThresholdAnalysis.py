import os
import numpy

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step8-Threshold/BothThreshold-OnlyIndividual-Number/'
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=int, delimiter=',')
        print('%d\t%d' % (data[0], data[1]))
