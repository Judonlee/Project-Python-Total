import os
import numpy
import matplotlib.pylab as plt
import pywt

WAVELET_CONF = 'db4'
if __name__ == '__main__':
    loadpath = 'E:/LIDC/TreatmentTrace/Step5-NonNodulesCsv/'
    csvSavepath = 'E:/LIDC/TreatmentTrace/Step6-Wavelet-NonNodules/%s_CSV/' % WAVELET_CONF
    pngSavepath = 'E:/LIDC/TreatmentTrace/Step6-Wavelet-NonNodules/%s_PNG/' % WAVELET_CONF
    for indexA in os.listdir(loadpath):
        print(indexA)

        if os.path.exists(os.path.join(csvSavepath, 'cA', indexA)): continue

        for part in ['cA', 'cH', 'cV', 'cD']:
            os.makedirs(os.path.join(csvSavepath, part, indexA))
            os.makedirs(os.path.join(pngSavepath, part, indexA))

        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            data = numpy.genfromtxt(fname=os.path.join(loadpath, indexA, indexB), dtype=int, delimiter=',')

            cA, (cH, cV, cD) = pywt.dwt2(data=data, wavelet=WAVELET_CONF)

            with open(os.path.join(csvSavepath, 'cA', indexA, indexB), 'w') as file:
                treatData = cA
                for indexX in range(numpy.shape(treatData)[0]):
                    for indexY in range(numpy.shape(treatData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(treatData[indexX][indexY]))
                    file.write('\n')
            with open(os.path.join(csvSavepath, 'cD', indexA, indexB), 'w') as file:
                treatData = cD
                for indexX in range(numpy.shape(treatData)[0]):
                    for indexY in range(numpy.shape(treatData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(treatData[indexX][indexY]))
                    file.write('\n')
            with open(os.path.join(csvSavepath, 'cH', indexA, indexB), 'w') as file:
                treatData = cH
                for indexX in range(numpy.shape(treatData)[0]):
                    for indexY in range(numpy.shape(treatData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(treatData[indexX][indexY]))
                    file.write('\n')
            with open(os.path.join(csvSavepath, 'cV', indexA, indexB), 'w') as file:
                treatData = cV
                for indexX in range(numpy.shape(treatData)[0]):
                    for indexY in range(numpy.shape(treatData)[1]):
                        if indexY != 0: file.write(',')
                        file.write(str(treatData[indexX][indexY]))
                    file.write('\n')

            plt.figure(figsize=(numpy.shape(cA)[0] / 100, numpy.shape(cA)[1] / 100))
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.imshow(cA, cmap='gray')
            plt.savefig(os.path.join(pngSavepath, 'cA', indexA, indexB + '.png'))
            plt.clf()
            plt.close()

            plt.figure(figsize=(numpy.shape(cD)[0] / 100, numpy.shape(cD)[1] / 100))
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.imshow(cD, cmap='gray')
            plt.savefig(os.path.join(pngSavepath, 'cD', indexA, indexB + '.png'))
            plt.clf()
            plt.close()

            plt.figure(figsize=(numpy.shape(cH)[0] / 100, numpy.shape(cH)[1] / 100))
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.imshow(cH, cmap='gray')
            plt.savefig(os.path.join(pngSavepath, 'cH', indexA, indexB + '.png'))
            plt.clf()
            plt.close()

            plt.figure(figsize=(numpy.shape(cV)[0] / 100, numpy.shape(cV)[1] / 100))
            plt.axis('off')
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.imshow(cV, cmap='gray')
            plt.savefig(os.path.join(pngSavepath, 'cV', indexA, indexB + '.png'))
            plt.clf()
            plt.close()

        # exit()
