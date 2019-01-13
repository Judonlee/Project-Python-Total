import numpy
import os

THRESHOLD = 32


def PositionAnalysis(filename):
    data = numpy.genfromtxt(filename, dtype=str, delimiter=',')
    pool = set()
    for sample in data:
        pool.add(sample[0])
    return len(pool)


def PositionAnalysisWithThreshold(filename, instanceNumberName, frontThreshold, backThreshold, flag=False):
    data = numpy.genfromtxt(filename, dtype=str, delimiter=',')
    dictionData = numpy.genfromtxt(instanceNumberName, dtype=str, delimiter=',')
    dictionary = {}
    for sample in dictionData:
        dictionary[sample[1]] = sample[0]

    pool = set()
    for sample in data:
        if sample[0] not in dictionary.keys():
            return 0
        pool.add(dictionary[sample[0]])
    # print(pool)

    zscope = []
    for sample in pool:
        zscope.append(int(sample))
    zscope = sorted(zscope)
    # print(zscope)
    # return len(pool)

    counter = 0
    if not flag:
        # print('HERE')
        startPosition = zscope[0] + (zscope[-1] - zscope[0]) * frontThreshold
        endPosition = zscope[-1] - (zscope[-1] - zscope[0]) * (1 - backThreshold)
        # print(startPosition, endPosition)

        for choose in zscope:
            if startPosition <= choose and choose <= endPosition:
                counter += 1
    else:
        # print('HERE')
        return max(len(pool) - 2 * frontThreshold, 0)
    return counter


if __name__ == '__main__':
    instancePath = 'E:/LIDC/TreatmentTrace/Step1-InstanceNumber/'
    characterPath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition/'
    noduleMediaPath = 'E:/LIDC/TreatmentTrace/Step3-NoduleMedia/'
    finalDecisionPath = 'E:/LIDC/TreatmentTrace/Step4-FinalDecision/'
    confName = 'SeparateConditionFlag'
    savePath = 'E:/LIDC/TreatmentTrace/Step8-Threshold/BothThreshold-%s-Number/' % confName
    if not os.path.exists(savePath): os.makedirs(savePath)

    for threshold in range(50):
        counter = 0
        dictionary = {}
        for one_use in os.listdir(instancePath):
            instanceName = one_use[0:one_use.find('.csv')]
            print('\rTreating \t %s' % instanceName, end='')

            if not os.path.exists(os.path.join(noduleMediaPath, instanceName + '.csv')): continue
            if not os.path.exists(os.path.join(finalDecisionPath, instanceName + '.csv')): continue

            noduleMediaText = numpy.reshape(
                numpy.genfromtxt(fname=os.path.join(noduleMediaPath, instanceName + '.csv'), dtype=str,
                                 delimiter=','), newshape=[-1, 4])
            finalDecisionText = numpy.reshape(
                numpy.genfromtxt(fname=os.path.join(finalDecisionPath, instanceName + '.csv'), dtype=str,
                                 delimiter=','), newshape=[-1, 5])

            flag = False
            for decisionNodule in finalDecisionText:
                if int(decisionNodule[-1]) < 2: continue
                for compareNodule in noduleMediaText:
                    distance = 0
                    for index in range(1, 4):
                        distance += abs(float(compareNodule[index]) - float(decisionNodule[index]))
                    if distance > THRESHOLD: continue

                    character = numpy.genfromtxt(
                        fname=os.path.join(characterPath, instanceName, compareNodule[0], confName + '.txt'),
                        dtype=int, delimiter=',')
                    if character == 1:
                        positionShape = PositionAnalysisWithThreshold(
                            filename=os.path.join(characterPath, instanceName, compareNodule[0], 'Position.csv'),
                            instanceNumberName=os.path.join(instancePath, instanceName + '.csv'),
                            frontThreshold=threshold, backThreshold=threshold, flag=True)
                        # exit()
                        counter += positionShape
                        if positionShape in dictionary:
                            dictionary[positionShape] += 1
                        else:
                            dictionary[positionShape] = 1
        with open(savePath + '%04d.csv' % threshold, 'w') as file:
            print('\n', threshold, counter)
            file.write(str(threshold) + ',' + str(counter))
