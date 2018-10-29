import numpy


def DXSingleCalculation(data, label):
    listZero, listOne = [], []
    for index in range(len(data)):
        if label[index] == 0: listZero.append(data[index])
        if label[index] == 1: listOne.append(data[index])
    # print(numpy.mean(listOne), numpy.mean(listZero))
    # print(numpy.std(listOne), numpy.std(listZero))
    DXScore = (numpy.mean(listOne) - numpy.mean(listZero)) * (numpy.mean(listOne) - numpy.mean(listZero)) / \
              (numpy.std(listOne) * numpy.std(listOne) + numpy.std(listZero) * numpy.std(listZero))
    return DXScore


def DXFeatureSelection(data, label, maxFeatures=10):
    totalScore = []
    for index in range(numpy.shape(data)[1]):
        print('\rTreating %d/%d' % (index, numpy.shape(data)[1]), end='')
        totalScore.append(DXSingleCalculation(data=data[:, index], label=label))

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
