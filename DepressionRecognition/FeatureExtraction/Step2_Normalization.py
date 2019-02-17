import os
import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/AVEC2017-Bands40/Step1_Npy/'
    savepath = 'D:/ProjectData/AVEC2017-Bands40/Step2_Csv_Normalization/'

    # loadpath = 'D:/ProjectData/AVEC2017-MFCC/Step1_OriginNpy/'
    # savepath = 'D:/ProjectData/AVEC2017-MFCC/Step2_Csv_Normalization/'

    totalData = []

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                data = numpy.load(os.path.join(loadpath, indexA, indexB, indexC))
                totalData.extend(data)
            print(indexA, indexB, numpy.shape(totalData))

    print(numpy.shape(totalData))
    totalData = scale(totalData)

    startPosition = 0
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            os.makedirs(os.path.join(savepath, indexA, indexB))
            print(indexA, indexB)
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                data = numpy.load(os.path.join(loadpath, indexA, indexB, indexC))
                with open(os.path.join(savepath, indexA, indexB, indexC[0:indexC.find('.')] + '.csv'), 'w') as file:
                    for indexX in range(len(data)):
                        for indexY in range(len(data[indexX])):
                            if indexY != 0: file.write(',')
                            file.write(str(totalData[indexX + startPosition][indexY]))
                        file.write('\n')
                # print(numpy.shape(data[startPosition:startPosition + len(data)]))
                # numpy.save(data[startPosition:startPosition + len(data)],
                #            os.path.join(savepath, indexA, indexB, indexC))

                startPosition += len(data)
