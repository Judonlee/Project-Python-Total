import os
import numpy
from skimage.feature import local_binary_pattern
import matplotlib.pylab as plt

POINT = 24
RADIUS = 3
if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step5-NonNodulesCsv/'
    csvSavepath = 'E:/LIDC/TreatmentTrace/Step6-LBP-NonNodules/P=%d_R=%d_CSV/' % (POINT, RADIUS)
    pngSavepath = 'E:/LIDC/TreatmentTrace/Step6-LBP-NonNodules/P=%d_R=%d_PNG/' % (POINT, RADIUS)
    for indexA in os.listdir(loadpath):
        if os.path.exists(os.path.join(csvSavepath, indexA)): continue
        os.makedirs(os.path.join(csvSavepath, indexA))
        os.makedirs(os.path.join(pngSavepath, indexA))
        print(indexA)

        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            # print(indexA, indexB)
            data = numpy.genfromtxt(fname=os.path.join(loadpath, indexA, indexB), dtype=int, delimiter=',')
            lbp = local_binary_pattern(image=data, P=POINT, R=RADIUS)

            with open(os.path.join(csvSavepath, indexA, indexB), 'w') as file:
                for indexX in range(numpy.shape(lbp)[0]):
                    for indexY in range(numpy.shape(lbp)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(lbp[indexX][indexY]))
                    file.write('\n')

            plt.figure(figsize=(numpy.shape(data)[0] / 100, numpy.shape(data)[1] / 100))
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.imshow(lbp, cmap='gray')
            plt.savefig(os.path.join(pngSavepath, indexA, indexB + '.png'))
            plt.clf()
            plt.close()
        # exit()
