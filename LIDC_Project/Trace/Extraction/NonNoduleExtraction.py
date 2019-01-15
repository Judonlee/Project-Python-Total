import os
import numpy
import pydicom
import matplotlib.pylab as plt

MULTI_DOCTOR_THRESHOLD = 2
DISTANCE_THRESHOLD = 32

if __name__ == '__main__':
    LIDCPath = 'E:/LIDC/LIDC-IDRI/'
    DictionaryPath = 'E:/LIDC/TreatmentTrace/Step0-Dictionary/'
    MediaPositionPath = 'E:/LIDC/TreatmentTrace/Step5-NonNodules-MediaPosition/'
    AssemblyPath = 'E:/LIDC/TreatmentTrace/Step6-NonNodules-Assembly/'
    SavePath = 'E:/LIDC/TreatmentTrace/FinalResult/NonNodules/'
    if not os.path.exists(SavePath): os.makedirs(SavePath)

    for InstanceName in os.listdir(LIDCPath):
        suppleNameA = os.listdir(os.path.join(LIDCPath, InstanceName))[0]
        suppleNameB = os.listdir(os.path.join(LIDCPath, InstanceName, suppleNameA))[0]

        print(InstanceName, suppleNameA, suppleNameB)

        ############################################################
        # Reassure

        if not os.path.exists(os.path.join(DictionaryPath, InstanceName + '.csv')): continue
        if not os.path.exists(os.path.join(MediaPositionPath, InstanceName + '.csv')): continue
        if not os.path.exists(os.path.join(AssemblyPath, InstanceName + '.csv')): continue

        ############################################################
        # Making Dictionary
        dictionary = {}
        dictionaryData = numpy.genfromtxt(fname=os.path.join(DictionaryPath, InstanceName + '.csv'), dtype=str,
                                          delimiter=',')
        for sample in dictionaryData:
            dictionary[int(sample[0])] = sample[2]

        ############################################################
        # Position Reading

        MediaPositionData = numpy.reshape(numpy.genfromtxt(
            fname=os.path.join(MediaPositionPath, InstanceName + '.csv'), dtype=int, delimiter=','), newshape=[-1, 3])
        AssemblyPositionData = numpy.reshape(numpy.genfromtxt(
            fname=os.path.join(AssemblyPath, InstanceName + '.csv'), dtype=int, delimiter=','), newshape=[-1, 4])

        if len(MediaPositionData) == 0 or len(AssemblyPositionData) == 0: continue
        ############################################################
        # Judgement

        CompareFlag = numpy.ones(len(MediaPositionData))
        for ChooseSample in AssemblyPositionData:
            if ChooseSample[-1] < MULTI_DOCTOR_THRESHOLD: continue

            for searchIndex in range(len(MediaPositionData)):
                CompareSample = MediaPositionData[searchIndex]
                Distance = numpy.sum(numpy.abs(ChooseSample[0:3] - CompareSample))
                if Distance > DISTANCE_THRESHOLD: continue
                if CompareFlag[searchIndex] != 1: continue
                CompareFlag[searchIndex] = 0

                ##########################################################
                # Judgement Completed

                DCMFile = pydicom.read_file(
                    fp=os.path.join(LIDCPath, InstanceName, suppleNameA, suppleNameB, dictionary[CompareSample[0]]))

                SaveCounter = 1
                if not os.path.exists(os.path.join(SavePath, InstanceName)):
                    os.makedirs(os.path.join(SavePath, InstanceName))
                while os.path.exists(os.path.join(SavePath, InstanceName, 'NonNodule-%04d.csv' % SaveCounter)):
                    SaveCounter += 1

                # print(SaveCounter)

                xPosition = CompareSample[1]
                yPosition = CompareSample[2]

                if xPosition > 512 - DISTANCE_THRESHOLD: xPosition = 512 - DISTANCE_THRESHOLD
                if yPosition > 512 - DISTANCE_THRESHOLD: yPosition = 512 - DISTANCE_THRESHOLD
                if xPosition < DISTANCE_THRESHOLD: xPosition = DISTANCE_THRESHOLD
                if yPosition < DISTANCE_THRESHOLD: yPosition = DISTANCE_THRESHOLD

                xPosition -= DISTANCE_THRESHOLD
                yPosition -= DISTANCE_THRESHOLD

                with open(os.path.join(SavePath, InstanceName, 'NonNodule-%04d.csv' % SaveCounter), 'w') as file:
                    for indexX in range(2 * DISTANCE_THRESHOLD):
                        for indexY in range(2 * DISTANCE_THRESHOLD):
                            if indexY != 0: file.write(',')
                            file.write(str(DCMFile.pixel_array[yPosition + indexX][xPosition + indexY]))
                        file.write('\n')
        # exit()
