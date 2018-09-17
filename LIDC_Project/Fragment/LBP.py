import os
import numpy
from skimage.feature import local_binary_pattern
import matplotlib.pylab as plt

if __name__ == '__main__':
    points, radius = 24, 3
    loadpath = 'E:\\BaiduNetdiskDownload\\Csv\\'
    savepath = 'E:\\BaiduNetdiskDownload\\LBP_Result_P=' + str(points) + '_R=' + str(radius) + '\\'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            pngPath = savepath + indexA + '\\' + indexB + '\\Png\\'
            csvPath = savepath + indexA + '\\' + indexB + '\\Csv\\'
            os.makedirs(pngPath)
            os.makedirs(csvPath)
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                print(indexA, indexB, indexC)

                treatData = numpy.genfromtxt(loadpath + indexA + '\\' + indexB + '\\' + indexC, dtype=int,
                                             delimiter=',')
                lbp = local_binary_pattern(image=treatData, P=points, R=radius)
                plt.figure(figsize=(0.64, 0.64))
                plt.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                # plt.imshow(treatData, cmap='gray')
                plt.savefig(pngPath + indexC + '.png')
                # plt.show()
                plt.clf()
                plt.close()

                file = open(csvPath + indexC, 'w')
                for indexX in range(len(lbp)):
                    for indexY in range(len(lbp[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(lbp[indexX][indexY]))
                    file.write('\n')
                file.close()

                # exit()
