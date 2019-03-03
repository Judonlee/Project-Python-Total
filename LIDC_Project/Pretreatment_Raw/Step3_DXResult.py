import numpy
from sklearn.decomposition import PCA


def DXSingleCalculation(data, label):
    listZero, listOne = [], []
    for index in range(len(data)):
        if label[index] == 2: listZero.append(data[index])
        if label[index] == 1: listOne.append(data[index])
    # print(numpy.mean(listOne), numpy.mean(listZero))
    # print(numpy.std(listOne), numpy.std(listZero))
    DXScore = (numpy.mean(listOne) - numpy.mean(listZero)) * (numpy.mean(listOne) - numpy.mean(listZero)) / \
              (numpy.std(listOne) * numpy.std(listOne) + numpy.std(listZero) * numpy.std(listZero))
    return DXScore


def DXFeatureSelection(data, label, maxFeatures=1):
    totalScore = []
    for index in range(numpy.shape(data)[1]):
        print('\rTreating %d/%d' % (index, numpy.shape(data)[1]), end='')
        score = DXSingleCalculation(data=data[:, index], label=label)
        # print('\n', score)
        totalScore.append(score)

    print('\n')
    featuresSelected = []
    for _ in range(maxFeatures):
        featuresSelected.append(numpy.argmax(totalScore))
        totalScore[numpy.argmax(totalScore)] = 0
    print(featuresSelected)

    results = []
    for indexX in range(numpy.shape(data)[0]):
        current = []
        for indexY in featuresSelected:
            current.append(data[indexX][indexY])
        results.append(current)
    print(numpy.shape(results))
    return results


if __name__ == '__main__':
    loadpath = 'E:/ProjectData_LIDC/Features/Step2_Features/CurveletFeature_%d.csv.npy'
    labelpath = 'E:/ProjectData_LIDC/Features/Step2_Features/Featurelabel_%d.csv'
    savepath = 'E:/ProjectData_LIDC/Features/Step3_DX/CurveletFeature_%d.csv.npy'

    totalData, totalLabel, totalThreshold = [], [], []
    for index in range(5):
        data = numpy.load(loadpath % index)
        label = numpy.genfromtxt(fname=labelpath % index, dtype=int, delimiter=',')

        totalData.extend(data)
        totalLabel.extend(label)

        totalThreshold.append(numpy.shape(data)[0])

    totalData = numpy.array(totalData)
    totalLabel = numpy.array(totalLabel)
    print(numpy.shape(totalData), numpy.shape(totalLabel))

    result = DXFeatureSelection(data=totalData, label=totalLabel, maxFeatures=30)
    print(numpy.shape(result))

    startPosition = 0
    for index in range(5):
        print(numpy.shape(result[startPosition:startPosition + totalThreshold[index]]))
        numpy.save(savepath % index, result[startPosition:startPosition + totalThreshold[index]])
        startPosition += totalThreshold[index]
