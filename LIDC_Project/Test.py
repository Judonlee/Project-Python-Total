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
    loadpath = 'E:/LIDC/LIDC-NonNodule-Wavelet/db4/Csv/'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                print(indexA, indexB, indexC)
                os.renames(old=os.path.join(loadpath, indexA, indexB, indexC),
                           new=os.path.join(loadpath, indexA, indexB, indexC[0:indexC.find('.')] + '.csv'))
                #exit()
                # for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                #     print(indexA, indexB, indexC, indexD)
                #     os.renames(old=os.path.join(loadpath, indexA, indexB, indexC, indexD),
                #                new=os.path.join(loadpath, indexA, indexB, indexC, indexD[0:indexD.find('.')] + '.csv'))
                #     #exit()
