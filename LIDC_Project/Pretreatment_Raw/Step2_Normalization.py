import numpy
import os
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/LIDC/LBP-Npy/R=3_P=24/'
    savepath = 'D:/LIDC/LBP-Npy/R=3_P=24_Normalization/'
    os.makedirs(savepath)

    totalData, totalThreshold = [], []
    for index in range(5):
        data = numpy.load(file=loadpath + 'Part%d-Data.npy' % index)
        data = numpy.reshape(data, [numpy.shape(data)[0], numpy.shape(data)[1] * numpy.shape(data)[2]])

        totalData.extend(data)
        totalThreshold.append(numpy.shape(data)[0])

    print(numpy.shape(totalData), totalThreshold)

    print(numpy.argwhere(numpy.isnan(totalData)))

    totalData = scale(totalData)

    startPosition = 0
    for index in range(len(totalThreshold)):
        print(numpy.shape(totalData[startPosition:startPosition + totalThreshold[index]]))
        numpy.save(file=savepath + 'Part%d-Data.npy' % index,
                   arr=totalData[startPosition:startPosition + totalThreshold[index]])
        startPosition += totalThreshold[index]
