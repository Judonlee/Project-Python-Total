import pywt
import os
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    biorFamily = ['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3',
                  'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8']
    coifFamily = ['coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11',
                  'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17']
    dbFamily = ['db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14',
                'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27',
                'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38']
    rbioFamily = ['rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3',
                  'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8']
    symFamily = ['sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13',
                 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']
    remainFamily = ['dmey', 'haar']
    
    for wavlet in biorFamily:
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
