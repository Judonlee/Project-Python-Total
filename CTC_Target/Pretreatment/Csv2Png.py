import numpy
import matplotlib.pylab as plt
import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Csv/Bands-100/'
    savepath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Png/Bands-100/'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            os.makedirs(os.path.join(savepath, indexA, indexB))
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                print(indexA, indexB, indexC)
                data = numpy.genfromtxt(fname=os.path.join(loadpath, indexA, indexB, indexC), dtype=float,
                                        delimiter=',')

                plt.figure(figsize=(numpy.shape(data)[1] / 100, numpy.shape(data)[0] / 100))
                plt.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.imshow(data)
                plt.savefig(os.path.join(savepath, indexA, indexB, indexC + '.current.png'))
                plt.clf()
                plt.close()

                data = plt.imread(os.path.join(savepath, indexA, indexB, indexC + '.current.png'))
                data = plt.resize(data, [227, 227])
                plt.figure(figsize=(numpy.shape(data)[1] / 100, numpy.shape(data)[0] / 100))
                plt.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.imshow(data)
                plt.savefig(os.path.join(savepath, indexA, indexB, indexC + '.png'))
                plt.clf()
                plt.close()

                os.remove(os.path.join(savepath, indexA, indexB, indexC + '.current.png'))
                # exit()
