import os
import matplotlib.pylab as plt
import numpy

if __name__ == '__main__':
    # loadpath = 'E:\\LIDC\\LIDC-NonNodules-CSV\\'
    # savepath = 'E:\\LIDC\\LIDC-NonNodules-PNG\\'
    # for indexA in os.listdir(loadpath):
    #     os.makedirs(savepath + indexA)
    #     for indexB in os.listdir(os.path.join(loadpath, indexA)):
    #         print(indexA, indexB)
    #         data = numpy.genfromtxt(loadpath + indexA + '\\' + indexB, dtype=float, delimiter=',')
    #         fig = plt.figure(figsize=(0.64, 0.64))
    #         plt.axis('off')
    #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #
    #         plt.imshow(data).set_cmap('gray')
    #         plt.savefig(savepath + indexA + '\\' + indexB + '.png')
    #         plt.clf()
    #         plt.close()
    print([1, 2, 3, 4, 5][0:5:2])
