import os
import numpy

if __name__ == '__main__':
    loadpath = 'F:\\AVEC-Final\\NetworkChangedAgain-SigmoidLastNot\\'
    for indexA in os.listdir(loadpath):
        totalData = []
        for indexB in os.listdir(loadpath + indexA):
            if indexB[-3:] != 'csv': continue
            currentData = numpy.genfromtxt(loadpath + indexA + '\\' + indexB, dtype=float, delimiter=',')
            totalData.append(currentData)
        print(indexA, end='\t')
        for sample in numpy.min(totalData, axis=0):
            print(sample, end='\t')
        print()
