import os
import numpy

THRESHOLD = 9

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step4-FinalDecision/'

    counter = 0
    for filename in os.listdir(loadpath):
        print(filename)
        with open(os.path.join(loadpath, filename), 'r') as file:
            data = file.readlines()
            for sample in data:
                sample = sample.split(',')
                if int(sample[-1]) > THRESHOLD: counter += 1
    print(counter)
