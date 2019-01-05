import numpy
import os

THRESHOLD = 32


def PositionAnalysis(filename):
    data = numpy.genfromtxt(filename, dtype=str, delimiter=',')
    pool = set()
    for sample in data:
        pool.add(sample[0])
    return len(pool)


if __name__ == '__main__':
    instancePath = 'E:/LIDC/TreatmentTrace/Step1-InstanceNumber/'
    characterPath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition/'
    noduleMediaPath = 'E:/LIDC/TreatmentTrace/Step3-NoduleMedia/'
    finalDecisionPath = 'E:/LIDC/TreatmentTrace/Step4-FinalDecision/'
    savePath = 'E:/LIDC/AnotherTrace/Step5-SeparateCondition/'

    counter = 0
    dictionary = {}
    for one_use in os.listdir(instancePath):
        instanceName = one_use[0:one_use.find('.csv')]
        print('Treating \t %s' % instanceName)

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
                    fname=os.path.join(characterPath, instanceName, compareNodule[0], 'SeparateConditionFlag.txt'),
                    dtype=int, delimiter=',')
                if character == 1:
                    positionShape = PositionAnalysis(
                        filename=os.path.join(characterPath, instanceName, compareNodule[0], 'Position.csv'))
                    counter += positionShape
                    if positionShape in dictionary:
                        dictionary[positionShape] += 1
                    else:
                        dictionary[positionShape] = 1
                    # counter += 1
                    # flag = True
        # if flag: counter += 1
    print(counter)

    # import matplotlib.pylab as plt
    #
    # # dictionary = [(k, dictionary[k]) for k in sorted(dictionary.keys())]
    #
    # newDictionary = []
    # for key in sorted(dictionary.keys()):
    #     newDictionary.append([key, dictionary[key]])
    # newDictionary = numpy.array(newDictionary)
    #
    # rect = plt.bar(newDictionary[:, 0], newDictionary[:, 1])
    # plt.xlabel('Nodule\'s Section Number')
    # plt.ylabel('Nodule Number')
    # plt.title('Separate Condition Nodule Number Analysis')
    # plt.show()
