import numpy
import os
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/FinalResult/Nodules/LIDC-IDRI-0001/Nodule-0001/'
    savepath = 'E:/LIDC/TreatmentTrace/FinalResult/Nodules/LIDC-IDRI-0001-Png/Nodule-0001/'
    if not os.path.exists(savepath): os.makedirs(savepath)
    for filename in os.listdir(loadpath):
        data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=int, delimiter=',')

        plt.figure(figsize=(numpy.shape(data)[0] / 100, numpy.shape(data)[1] / 100))
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.imshow(data, cmap='gray')
        plt.savefig(os.path.join(savepath, filename + '.png'))
        plt.clf()
        plt.close()
