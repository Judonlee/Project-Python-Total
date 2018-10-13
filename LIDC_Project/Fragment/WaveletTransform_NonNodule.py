import pywt
import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    dbFamily = ['db1', 'db2', 'db4']

    for wavlet in ['db4']:
        loadpath = 'E:\\LIDC\\LIDC-NonNodules-CSV\\'
        savepath = 'E:\\LIDC\\LIDC-NonNodule-Wavelet\\' + wavlet + '\\'
        pngPath = savepath + 'Png\\'
        csvPath = savepath + 'Csv\\'

        for indexA in os.listdir(loadpath):
            for indexB in os.listdir(loadpath + indexA):
                print(indexA, indexB)
                if not os.path.exists(os.path.join(pngPath, indexA, indexB)):
                    os.makedirs(os.path.join(pngPath, indexA, indexB))
                    os.makedirs(os.path.join(csvPath, indexA, indexB))

                treatData = numpy.genfromtxt(loadpath + indexA + '\\' + indexB, dtype=int, delimiter=',')
                cA, (cH, cV, cD) = pywt.dwt2(data=treatData, wavelet=wavlet)

                plt.figure(figsize=(numpy.shape(cA)[0] / 100, numpy.shape(cA)[1] / 100))
                plt.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.imshow(cA, cmap='gray')
                plt.savefig(os.path.join(pngPath, indexA, indexB, 'cA.png'))
                plt.clf()
                plt.close()

                plt.figure(figsize=(numpy.shape(cH)[0] / 100, numpy.shape(cH)[1] / 100))
                plt.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.imshow(cH, cmap='gray')
                plt.savefig(os.path.join(pngPath, indexA, indexB, 'cH.png'))
                plt.clf()
                plt.close()

                plt.figure(figsize=(numpy.shape(cV)[0] / 100, numpy.shape(cV)[1] / 100))
                plt.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.imshow(cV, cmap='gray')
                plt.savefig(os.path.join(pngPath, indexA, indexB, 'cV.png'))
                plt.clf()
                plt.close()

                plt.figure(figsize=(numpy.shape(cD)[0] / 100, numpy.shape(cD)[1] / 100))
                plt.axis('off')
                plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                plt.imshow(cD, cmap='gray')
                plt.savefig(os.path.join(pngPath, indexA, indexB, 'cD.png'))
                plt.clf()
                plt.close()

                file = open(os.path.join(csvPath, indexA, indexB, 'cA.png'), 'w')
                for indexX in range(len(cA)):
                    for indexY in range(len(cA[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(cA[indexX][indexY]))
                    file.write('\n')
                file.close()

                file = open(os.path.join(csvPath, indexA, indexB, 'cH.png'), 'w')
                for indexX in range(len(cH)):
                    for indexY in range(len(cH[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(cH[indexX][indexY]))
                    file.write('\n')
                file.close()

                file = open(os.path.join(csvPath, indexA, indexB, 'cV.png'), 'w')
                for indexX in range(len(cV)):
                    for indexY in range(len(cV[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(cV[indexX][indexY]))
                    file.write('\n')
                file.close()

                file = open(os.path.join(csvPath, indexA, indexB, 'cD.png'), 'w')
                for indexX in range(len(cD)):
                    for indexY in range(len(cD[indexX])):
                        if indexY != 0: file.write(',')
                        file.write(str(cD[indexX][indexY]))
                    file.write('\n')
                file.close()
