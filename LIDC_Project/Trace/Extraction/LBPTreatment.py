import os
from skimage.feature import local_binary_pattern
import numpy
import matplotlib.pylab as plt
import pywt

POINT = 24
RADIUS = 3
WAVELET_CONF = 'db4'


def LBPTreatment(loadpath, savepathCsv, savepathPng, radius, point):
    data = numpy.genfromtxt(fname=loadpath, dtype=int, delimiter=',')
    # print(numpy.shape(data))
    data = local_binary_pattern(image=data, P=point, R=radius)

    plt.figure(figsize=(numpy.shape(data)[0] / 100, numpy.shape(data)[1] / 100))
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(data, cmap='gray')
    plt.savefig(savepathPng + '.png')
    # plt.show()
    plt.clf()
    plt.close()

    with open(savepathCsv + '.csv', 'w') as file:
        for indexX in range(numpy.shape(data)[0]):
            for indexY in range(numpy.shape(data)[1]):
                if indexY != 0: file.write(',')
                file.write(str(data[indexX][indexY]))
            file.write('\n')


def WaveletTreatment(loadpath, savepathCsv, savepathPng, conf):
    data = numpy.genfromtxt(fname=loadpath, dtype=int, delimiter=',')
    cA, (cH, cV, cD) = pywt.dwt2(data=data, wavelet=conf)

    with open(savepathCsv + '-cA.csv', 'w') as file:
        treatData = cA
        for indexX in range(numpy.shape(treatData)[0]):
            for indexY in range(numpy.shape(treatData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(treatData[indexX][indexY]))
            file.write('\n')
    with open(savepathCsv + '-cD.csv', 'w') as file:
        treatData = cD
        for indexX in range(numpy.shape(treatData)[0]):
            for indexY in range(numpy.shape(treatData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(treatData[indexX][indexY]))
            file.write('\n')
    with open(savepathCsv + '-cH.csv', 'w') as file:
        treatData = cH
        for indexX in range(numpy.shape(treatData)[0]):
            for indexY in range(numpy.shape(treatData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(treatData[indexX][indexY]))
            file.write('\n')
    with open(savepathCsv + '-cV.csv', 'w') as file:
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
    plt.savefig(savepathPng + '-cA.png')
    plt.clf()
    plt.close()

    plt.figure(figsize=(numpy.shape(cA)[0] / 100, numpy.shape(cA)[1] / 100))
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(cD, cmap='gray')
    plt.savefig(savepathPng + '-cD.png')
    plt.clf()
    plt.close()

    plt.figure(figsize=(numpy.shape(cA)[0] / 100, numpy.shape(cA)[1] / 100))
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(cH, cmap='gray')
    plt.savefig(savepathPng + '-cH.png')
    plt.clf()
    plt.close()

    plt.figure(figsize=(numpy.shape(cA)[0] / 100, numpy.shape(cA)[1] / 100))
    plt.axis('off')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.imshow(cV, cmap='gray')
    plt.savefig(savepathPng + '-cV.png')
    plt.clf()
    plt.close()

    # exit()


def FoldSearcher(loadpath, savepathCsv, savepathPng):
    for filename in os.listdir(loadpath):
        if os.path.isdir(os.path.join(loadpath, filename)):
            os.makedirs(os.path.join(savepathCsv, filename))
            os.makedirs(os.path.join(savepathPng, filename))
            FoldSearcher(loadpath=os.path.join(loadpath, filename), savepathPng=os.path.join(savepathPng, filename),
                         savepathCsv=os.path.join(savepathCsv, filename))
        else:
            # LBPTreatment(loadpath=)
            print(loadpath, savepathCsv)
            # LBPTreatment(loadpath=os.path.join(loadpath, filename), savepathCsv=os.path.join(savepathCsv, filename),
            #              savepathPng=os.path.join(savepathPng, filename), radius=RADIUS, point=POINT)
            WaveletTreatment(loadpath=os.path.join(loadpath, filename), savepathCsv=os.path.join(savepathCsv, filename),
                             savepathPng=os.path.join(savepathPng, filename), conf=WAVELET_CONF)


if __name__ == '__main__':
    for part in range(5):
        loadpath = 'D:/LIDC/Part%d/' % part
        savepathPng = 'D:/LIDC/Wavelet/%s_Png/Part%d/' % (WAVELET_CONF, part)
        savepathCsv = 'D:/LIDC/Wavelet/%s_Csv/Part%d/' % (WAVELET_CONF, part)
        FoldSearcher(loadpath=loadpath, savepathCsv=savepathCsv, savepathPng=savepathPng)
        # exit()
