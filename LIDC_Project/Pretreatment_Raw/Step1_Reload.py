import numpy
import os

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_LIDC/Features/Step0_Raw/'
    savepath = 'E:/ProjectData_LIDC/Features/Step1_Npy-Media/'
    if not os.path.exists(savepath): os.makedirs(savepath)

    for filename in os.listdir(loadpath):
        if filename.find('label') != -1: continue
        print(filename)
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',')
        print(max(data), min(data))
        # numpy.save(file=os.path.join(savepath, filename[0:filename.find('.')] + '.csv'), arr=data)
