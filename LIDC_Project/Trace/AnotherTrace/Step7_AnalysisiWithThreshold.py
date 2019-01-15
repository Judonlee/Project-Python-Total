import os
import numpy

THRESHOLD = 32


def PositionAnalysis(filename):
    data = numpy.genfromtxt(filename, dtype=str, delimiter=',')
    pool = set()
    for sample in data:
        pool.add(sample[0])
    return len(pool)


if __name__ == '__main__':
    decisionName = 'AtLeastTwo.txt'
    nodulePath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition/'
    mediaPositionPath = 'E:/LIDC/TreatmentTrace/Step3-NoduleMedia/'
    finalDecisionPath = 'E:/LIDC/TreatmentTrace/Step4-FinalDecision/'

    dictionary = {}
    for indexA in os.listdir(finalDecisionPath):
        print(indexA)
        instanceName = indexA[0:indexA.find('.')]
        mediaPositionData = numpy.genfromtxt(fname=os.path.join(mediaPositionPath, instanceName + '.csv'), dtype=str,
                                             delimiter=',')
        finalDecisionData = numpy.genfromtxt(fname=os.path.join(finalDecisionPath, instanceName + '.csv'), dtype=str,
                                             delimiter=',')

        if len(numpy.shape(mediaPositionData)) < 1: continue
        if len(numpy.shape(finalDecisionData)) < 1: continue

        mediaPositionData = numpy.reshape(mediaPositionData, [-1, 4])
        finalDecisionData = numpy.reshape(finalDecisionData, [-1, 5])
        for chooseNodule in finalDecisionData:
            counter = 0
            if int(chooseNodule[-1]) < 2: continue
            for compareNodule in mediaPositionData:
                distance = abs(float(compareNodule[1]) - float(chooseNodule[1])) + abs(
                    float(compareNodule[2]) - float(chooseNodule[2])) + abs(
                    float(compareNodule[3]) - float(chooseNodule[3]))
                if distance > THRESHOLD: continue

                #############################################################

                # print(compareNodule)
                if numpy.genfromtxt(fname=os.path.join(nodulePath,instanceName,compareNodule[0],decisionName),dtype=int,delimiter=',')==1:
                    SectionNumber = PositionAnalysis(
                        os.path.join(nodulePath, instanceName, compareNodule[0], 'Position.csv'))
                # print(SectionNumber)
                    counter += SectionNumber
            if counter in dictionary.keys():
                dictionary[counter] += 1
            else:
                dictionary[counter] = 1
        # exit()
    for sample in dictionary.keys():
        print(sample, '\t', dictionary[sample])
