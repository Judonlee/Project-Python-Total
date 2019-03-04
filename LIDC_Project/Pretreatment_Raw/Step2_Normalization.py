import numpy
import os
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_LIDC/Features/Step1_Npy-Media/DicFeature_Restart_%d.npy'
    savepath = 'E:/ProjectData_LIDC/Features/Step2_Features/DicFeature_Restart_%d.npy'
    # os.makedirs(savepath)

    totalData, totalThreshold = [], []
    for index in range(5):
        data = numpy.load(file=loadpath % index)

        totalData.extend(data)
        totalThreshold.append(numpy.shape(data)[0])

    print(numpy.shape(totalData), totalThreshold)

    print(numpy.argwhere(numpy.isnan(totalData)))

    totalData = scale(totalData)

    startPosition = 0
    for index in range(len(totalThreshold)):
        print(numpy.shape(totalData[startPosition:startPosition + totalThreshold[index]]))
        numpy.save(file=savepath % index, arr=totalData[startPosition:startPosition + totalThreshold[index]])
        startPosition += totalThreshold[index]
