import numpy
from sklearn.decomposition import PCA
import os

if __name__ == '__main__':
    loadpath = 'D:/LIDC/LBP-Npy/R=3_P=24_Normalization/Part%d-Data.npy'
    os.makedirs('D:/LIDC/LBP-Npy/R=3_P=24_PCA')
    savepath = 'D:/LIDC/LBP-Npy/R=3_P=24_PCA/Part%d-Data.npy'

    totalData, totalThreshold = [], []
    for index in range(5):
        data = numpy.load(loadpath % index)
        totalData.extend(data)
        totalThreshold.append(numpy.shape(data)[0])
    print(numpy.shape(totalData))

    pca = PCA(n_components=30)
    pca.fit(totalData)
    print(sum(pca.explained_variance_ratio_))
    result = pca.transform(totalData)

    startPosition = 0
    for index in range(5):
        print(numpy.shape(result[startPosition:startPosition + totalThreshold[index]]))
        numpy.save(savepath % index, result[startPosition:startPosition + totalThreshold[index]])
        startPosition += totalThreshold[index]
