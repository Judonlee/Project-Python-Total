import numpy
from sklearn.decomposition import PCA

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_LIDC/Features/Step2_Features/CurveletFeature_%d.csv.npy'
    savepath = 'E:/ProjectData_LIDC/Features/Step3_PCA/CurveletFeature_%d.csv.npy'

    totalData, totalThreshold = [], []
    for index in range(5):
        data = numpy.load(loadpath % index)
        totalData.extend(data)
        totalThreshold.append(numpy.shape(data)[0])
    print(numpy.shape(totalData))

    pca = PCA(n_components=200)
    pca.fit(totalData)
    result = pca.transform(totalData)

    startPosition = 0
    for index in range(5):
        print(numpy.shape(result[startPosition:startPosition + totalThreshold[index]]))
        numpy.save(savepath % index, result[startPosition:startPosition + totalThreshold[index]])
        startPosition += totalThreshold[index]
