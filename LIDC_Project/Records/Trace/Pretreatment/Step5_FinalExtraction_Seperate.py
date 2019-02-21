import os
import numpy
import pydicom
import matplotlib.pylab as plt

DECISION_THRESHOLD = 2
DISTANCE_THRESHOLD = 32
FIGURE_SCOPE = 32

if __name__ == '__main__':
    LIDCPath = 'E:/LIDC/LIDC-IDRI/'
    NoduleRecordsPath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition/'
    NoduleMediaPath = 'E:/LIDC/TreatmentTrace/Step3-NoduleMedia/'
    FinalDecisionPath = 'E:/LIDC/TreatmentTrace/Step4-FinalDecision/'
    SavePath = 'E:/LIDC/TreatmentTrace/Step5-NodulesCsv-Seperate/'
    for InstanceName in os.listdir(FinalDecisionPath):
        if os.path.exists(os.path.join(SavePath, InstanceName)): continue
        os.makedirs(os.path.join(SavePath, InstanceName))

        ######################################################################################

        filenameDictionary = {}
        indexA = os.listdir(os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')]))[0]
        indexB = os.listdir(os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')], indexA))[0]
        for indexC in os.listdir(os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')], indexA, indexB)):
            if indexC[-3:] != 'dcm': continue
            # print(indexC)
            DCMFile = pydicom.read_file(
                os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')], indexA, indexB, indexC))
            filenameDictionary[DCMFile.SOPInstanceUID] = indexC

        # print(InstanceName, filenameDictionary)
        ######################################################################################

        with open(os.path.join(FinalDecisionPath, InstanceName), 'r') as file:
            data = file.readlines()

        decisionNodulesDetails = []

        for sample in data:
            decisionNodulesDetails.append(sample[0:-1].split(','))
            for index in range(1, len(decisionNodulesDetails[-1])):
                decisionNodulesDetails[-1][index] = float(decisionNodulesDetails[-1][index])

        noduleRecords = set()
        for treatNodule in decisionNodulesDetails:
            if treatNodule[-1] < DECISION_THRESHOLD: continue
            print(InstanceName, treatNodule)

            noduleMediaData = numpy.genfromtxt(fname=os.path.join(NoduleMediaPath, InstanceName), dtype=str,
                                               delimiter=',')
            for sample in noduleMediaData:
                currentPosition = []
                for index in range(1, len(sample)):
                    currentPosition.append(float(sample[index]))

                distance = numpy.sum(numpy.abs(numpy.subtract(currentPosition, treatNodule[1:-1])))
                if distance < DISTANCE_THRESHOLD:
                    noduleRecords.add(sample[0])
        # print(noduleRecords)

        for chooseNodule in noduleRecords:
            if os.path.exists(os.path.join(SavePath, InstanceName, chooseNodule)): continue
            os.makedirs(os.path.join(SavePath, InstanceName, chooseNodule))
            # print(chooseNodule)

            nodulePosition = numpy.genfromtxt(
                fname=os.path.join(NoduleRecordsPath, InstanceName[0:InstanceName.find('.')], chooseNodule,
                                   'Position.csv'), dtype=str, delimiter=',')
            # print(nodulePosition)
            chooseNoduleDictionary = set()
            for sample in nodulePosition:
                chooseNoduleDictionary.add(sample[0])
            # print(chooseNoduleDictionary)

            for UID in chooseNoduleDictionary:
                dcmPosition = []
                for sample in nodulePosition:
                    if sample[0] == UID:
                        dcmPosition.append([float(sample[1]), float(sample[2])])
                [yPosition, xPosition] = numpy.median(dcmPosition, axis=0)

                xPosition = int(xPosition)
                yPosition = int(yPosition)

                if xPosition < FIGURE_SCOPE: xPosition = FIGURE_SCOPE
                if xPosition > 512 - FIGURE_SCOPE: xPosition = 512 - FIGURE_SCOPE
                if yPosition < FIGURE_SCOPE: yPosition = FIGURE_SCOPE
                if yPosition > 512 - FIGURE_SCOPE: yPosition = 512 - FIGURE_SCOPE
                xPosition -= FIGURE_SCOPE
                yPosition -= FIGURE_SCOPE

                if UID not in filenameDictionary.keys(): continue
                DCMFile = pydicom.read_file(
                    os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')], indexA, indexB,
                                 filenameDictionary[UID]))
                # print(xPosition, yPosition)

                # plt.imshow(
                #     DCMFile.pixel_array[xPosition:xPosition + 2 * FIGURE_SCOPE, yPosition:yPosition + 2 * FIGURE_SCOPE])
                # plt.show()

                fileCounter = 0
                while True:
                    fileCounter += 1
                    if not os.path.exists(
                            os.path.join(SavePath, InstanceName, chooseNodule, 'Part%04d.csv' % fileCounter)):
                        break

                with open(
                        os.path.join(SavePath, InstanceName, chooseNodule, 'Part%04d.csv' % fileCounter),
                        'w') as file:
                    for indexX in range(2 * FIGURE_SCOPE):
                        for indexY in range(2 * FIGURE_SCOPE):
                            if indexY != 0: file.write(',')
                            file.write(str(DCMFile.pixel_array[indexX + xPosition, indexY + yPosition]))
                        file.write('\n')
            # exit()

        # exit()
