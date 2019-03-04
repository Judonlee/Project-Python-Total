import numpy
import os

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_LIDC/Features/Step0_Raw/DicFeature_Restart_%d.csv'
    savepath = 'E:/ProjectData_LIDC/Features/Step1_Npy-Media/DicFeature_Restart_%d.npy'
    # if not os.path.exists(savepath): os.makedirs(savepath)

    for index in range(5):
        data = numpy.genfromtxt(fname=loadpath % index, dtype=float, delimiter=',')
        print(numpy.shape(data))
        numpy.save(file=savepath % index, arr=data)
