import os
import numpy
import pydicom

DISTANCE_THRESHOLD = 32
CHOOSE_CONF = 'AtLeastTwo.txt'

if __name__ == '__main__':
    LIDCPath = 'E:/LIDC/LIDC-IDRI/'
    DictionaryPath = 'E:/LIDC/TreatmentTrace/Step0-Dictionary/'
    MediaPositionPath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition/'
    SavePath = 'E:/LIDC/TreatmentTrace/FinalResult/Nodules/'
    if not os.path.exists(SavePath): os.makedirs(SavePath)

    for InstanceName in os.listdir(LIDCPath):
        suppleNameA = os.listdir(os.path.join(LIDCPath, InstanceName))[0]
        suppleNameB = os.listdir(os.path.join(LIDCPath, InstanceName, suppleNameA))[0]

        print(InstanceName, suppleNameA, suppleNameB)

        if not os.path.exists(os.path.join(DictionaryPath, InstanceName + '.csv')): continue

        ############################################################
        # Making Dictionary
        dictionaryDOI2Number, dictionaryNumber2Name = {}, {}
        dictionaryData = numpy.genfromtxt(fname=os.path.join(DictionaryPath, InstanceName + '.csv'), dtype=str,
                                          delimiter=',')
        for sample in dictionaryData:
            dictionaryDOI2Number[sample[1]] = int(sample[0])
        for sample in dictionaryData:
            dictionaryNumber2Name[int(sample[0])] = sample[2]

        ############################################################

        for NoduleName in os.listdir(os.path.join(MediaPositionPath, InstanceName)):
            if not os.path.exists(os.path.join(MediaPositionPath, InstanceName, NoduleName, CHOOSE_CONF)): continue
            if numpy.genfromtxt(fname=os.path.join(MediaPositionPath, InstanceName, NoduleName, CHOOSE_CONF), dtype=int,
                                delimiter=',') == 1:
                position = numpy.genfromtxt(
                    fname=os.path.join(MediaPositionPath, InstanceName, NoduleName, 'Position.csv'), dtype=str,
                    delimiter=',')

                positionTransform = []
                SectionSet = set()

                for sample in position:
                    if sample[0] not in dictionaryDOI2Number.keys(): continue
                    positionTransform.append([dictionaryDOI2Number[sample[0]], int(sample[1]), int(sample[2])])
                    SectionSet.add(dictionaryDOI2Number[sample[0]])

                result = sorted(SectionSet)
                for index in range(1, len(result) - 1):
                    xPosition, yPosition = [], []
                    for sample in positionTransform:
                        if sample[0] == result[index]:
                            xPosition.append(sample[1])
                            yPosition.append(sample[2])

                    finalXPosition = int(numpy.median(xPosition))
                    finalYPosition = int(numpy.median(yPosition))

                    if finalXPosition > 512 - DISTANCE_THRESHOLD: finalXPosition = 512 - DISTANCE_THRESHOLD
                    if finalYPosition > 512 - DISTANCE_THRESHOLD: finalYPosition = 512 - DISTANCE_THRESHOLD
                    if finalXPosition < DISTANCE_THRESHOLD: finalXPosition = DISTANCE_THRESHOLD
                    if finalYPosition < DISTANCE_THRESHOLD: finalYPosition = DISTANCE_THRESHOLD

                    finalXPosition -= DISTANCE_THRESHOLD
                    finalYPosition -= DISTANCE_THRESHOLD

                    DCMFile = pydicom.read_file(
                        fp=os.path.join(LIDCPath, InstanceName, suppleNameA, suppleNameB,
                                        dictionaryNumber2Name[result[index]]))

                    if not os.path.exists(os.path.join(SavePath, InstanceName, NoduleName)):
                        os.makedirs(os.path.join(SavePath, InstanceName, NoduleName))

                    SaveName = 1
                    while os.path.exists(
                            os.path.join(SavePath, InstanceName, NoduleName, 'Part-%04d.csv' % SaveName)):
                        SaveName += 1

                    with open(os.path.join(SavePath, InstanceName, NoduleName, 'Part-%04d.csv' % SaveName),
                              'w') as file:
                        for indexX in range(2 * DISTANCE_THRESHOLD):
                            for indexY in range(2 * DISTANCE_THRESHOLD):
                                if indexY != 0: file.write(',')
                                file.write(str(DCMFile.pixel_array[finalYPosition + indexX][finalXPosition + indexY]))
                            file.write('\n')

        # exit()
