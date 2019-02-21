import os
import numpy

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition/'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            dictionary = set()
            data = numpy.genfromtxt(fname=os.path.join(loadpath, indexA, indexB, 'Position.csv'), dtype=str,
                                    delimiter=',')
            for sample in data:
                dictionary.add(sample[0])
            print(indexA, indexB, len(dictionary))
            with open(os.path.join(loadpath, indexA, indexB, 'Fragment.csv'), 'w') as file:
                file.write(str(len(dictionary)))
            # print(indexA, indexB)
