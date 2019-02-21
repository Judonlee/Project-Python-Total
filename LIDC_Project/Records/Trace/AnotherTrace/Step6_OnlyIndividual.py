import numpy
import os

THRESHOLD = 32

if __name__ == '__main__':
    instancePath = 'E:/LIDC/TreatmentTrace/Step1-InstanceNumber/'
    characterPath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition/'
    noduleMediaPath = 'E:/LIDC/TreatmentTrace/Step3-NoduleMedia/'
    finalDecisionPath = 'E:/LIDC/TreatmentTrace/Step4-FinalDecision/'
    savePath = 'E:/LIDC/AnotherTrace/Step5-SeparateCondition/'

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

        for decisionNodule in finalDecisionText:
            for compareNodule in noduleMediaText:
                distance = 0
                for index in range(1, 4):
                    distance += abs(float(compareNodule[index]) - float(decisionNodule[index]))
                if distance > THRESHOLD: continue

                character = numpy.genfromtxt(
                    fname=os.path.join(characterPath, instanceName, compareNodule[0], 'CharacterDecision.txt'),
                    dtype=int, delimiter=',')

                flag = (character[0] == 1) and (character[1] == 1) and (character[2] == 1)
                with open(os.path.join(characterPath, instanceName, compareNodule[0], 'OnlyIndividual.txt'),
                          'w') as file:
                    file.write(str(int(flag)))
        #     exit()
        # exit()
