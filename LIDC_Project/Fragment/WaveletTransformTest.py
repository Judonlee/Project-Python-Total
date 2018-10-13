import pywt
import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    dbFamily = ['db1', 'db2', 'db4']

    for wavlet in dbFamily:
        loadpath = 'D:\\ProjectData\\Csv\\'
        for indexA in os.listdir(loadpath):
            for indexB in os.listdir(loadpath + indexA):
                pngSavepath = 'F:\\WaveletTransform-Again\\' + wavlet + '-png\\' + indexA + '\\' + indexB + '\\'
                if os.path.exists(pngSavepath): continue

                os.makedirs(pngSavepath + 'cA')
                os.makedirs(pngSavepath + 'cH')
                os.makedirs(pngSavepath + 'cV')
                os.makedirs(pngSavepath + 'cD')

                csvSavePath = 'F:\\WaveletTransform-Again\\' + wavlet + '-csv\\' + indexA + '\\' + indexB + '\\'
                os.makedirs(csvSavePath + 'cA')
                os.makedirs(csvSavePath + 'cH')
                os.makedirs(csvSavePath + 'cV')
                os.makedirs(csvSavePath + 'cD')

                for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                    print(wavlet, indexA, indexB, indexC)
                    treatData = numpy.genfromtxt(loadpath + indexA + '\\' + indexB + '\\' + indexC, dtype=int,
                                                 delimiter=',')
                    cA, (cH, cV, cD) = pywt.dwt2(data=treatData, wavelet=wavlet)

                    plt.figure(figsize=(numpy.shape(cA)[0] / 100, numpy.shape(cA)[1] / 100))
                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.imshow(cA, cmap='gray')
                    plt.savefig(pngSavepath + 'cA\\' + indexC + '.png')
                    plt.clf()
                    plt.close()

                    plt.figure(figsize=(numpy.shape(cH)[0] / 100, numpy.shape(cH)[1] / 100))
                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.imshow(cH, cmap='gray')
                    plt.savefig(pngSavepath + 'cH\\' + indexC + '.png')
                    plt.clf()
                    plt.close()

                    plt.figure(figsize=(numpy.shape(cV)[0] / 100, numpy.shape(cV)[1] / 100))
                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.imshow(cV, cmap='gray')
                    plt.savefig(pngSavepath + 'cV\\' + indexC + '.png')
                    plt.clf()
                    plt.close()

                    plt.figure(figsize=(numpy.shape(cD)[0] / 100, numpy.shape(cD)[1] / 100))
                    plt.axis('off')
                    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                    plt.imshow(cD, cmap='gray')
                    plt.savefig(pngSavepath + 'cD\\' + indexC + '.png')
                    plt.clf()
                    plt.close()

                    file = open(csvSavePath + 'cA\\' + indexC, 'w')
                    for indexX in range(len(cA)):
                        for indexY in range(len(cA[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(cA[indexX][indexY]))
                        file.write('\n')
                    file.close()

                    file = open(csvSavePath + 'cH\\' + indexC, 'w')
                    for indexX in range(len(cH)):
                        for indexY in range(len(cH[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(cH[indexX][indexY]))
                        file.write('\n')
                    file.close()

                    file = open(csvSavePath + 'cV\\' + indexC, 'w')
                    for indexX in range(len(cV)):
                        for indexY in range(len(cV[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(cV[indexX][indexY]))
                        file.write('\n')
                    file.close()

                    file = open(csvSavePath + 'cD\\' + indexC, 'w')
                    for indexX in range(len(cD)):
                        for indexY in range(len(cD[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(cD[indexX][indexY]))
                        file.write('\n')
                    file.close()
