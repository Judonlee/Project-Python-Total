import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/GitHub/CTC_Project_Again/Train/Results-Decode/Part5/'
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=loadpath + filename, dtype=float, delimiter=',')
        print(data[0], ',', data[1])
