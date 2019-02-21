import os
import numpy
from skimage.feature import local_binary_pattern
import matplotlib.pylab as plt

if __name__ == '__main__':
    points, radius = 24, 3
    loadpath = 'E:\\LIDC\\LIDC-Nodules-Selected\\'
    savepath = 'E:\\LIDC\\LIDC-Nodules-LBP\\Result_P=' + str(points) + '_R=' + str(radius) + '\\'

    pngPath = savepath + 'Png\\'
    csvPath = savepath + 'Csv\\'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            os.makedirs(pngPath + indexA + '\\' + indexB)
            os.makedirs(csvPath + indexA + '\\' + indexB)
            for indexC in ['Csv']:
                for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                    print(indexA, indexB, indexC, indexD)
                    treatData = numpy.genfromtxt(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD,
                                                 dtype=int, delimiter=',')
                    lbp = local_binary_pattern(image=treatData, P=points, R=radius)
                    plt.figure(figsize=(0.64, 0.64))
                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.imshow(lbp, cmap='gray')
                    plt.savefig(pngPath + indexA + '\\' + indexB + '\\' + indexD + '.png')
                    plt.clf()
                    plt.close()

                    file = open(csvPath + indexA + '\\' + indexB + '\\' + indexD, 'w')
                    for indexX in range(len(lbp)):
                        for indexY in range(len(lbp[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(lbp[indexX][indexY]))
                        file.write('\n')
                    file.close()
