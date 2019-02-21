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
    SavePath = 'E:/LIDC/TreatmentTrace/Step5-NodulesCsv/'
    for InstanceName in os.listdir(FinalDecisionPath):
        if os.path.exists(os.path.join(SavePath, InstanceName)): continue
        with open(os.path.join(FinalDecisionPath, InstanceName), 'r') as file:
            data = file.readlines()

        decisionNodulesDetails = []

        for sample in data:
            decisionNodulesDetails.append(sample[0:-1].split(','))
            for index in range(1, len(decisionNodulesDetails[-1])):
                decisionNodulesDetails[-1][index] = float(decisionNodulesDetails[-1][index])

        for treatNodule in decisionNodulesDetails:
            noduleRecords = []

            if treatNodule[-1] < DECISION_THRESHOLD: continue
            if os.path.exists(os.path.join(SavePath, InstanceName, treatNodule[0])): continue
            os.makedirs(os.path.join(SavePath, InstanceName, treatNodule[0]))
            print(InstanceName, treatNodule)

            noduleMediaData = numpy.genfromtxt(fname=os.path.join(NoduleMediaPath, InstanceName), dtype=str,
                                               delimiter=',')
            for sample in noduleMediaData:
                currentPosition = []
                for index in range(1, len(sample)):
                    currentPosition.append(float(sample[index]))

                distance = numpy.sum(numpy.abs(numpy.subtract(currentPosition, treatNodule[1:-1])))
                if distance < DISTANCE_THRESHOLD:
                    noduleRecords.append(sample[0])
            # print(noduleRecords)

            positionRecords = []
            positionDOISet = {}
            for noduleName in noduleRecords:
                nodulePosition = numpy.genfromtxt(
                    fname=os.path.join(NoduleRecordsPath, InstanceName[0:InstanceName.find('.')], noduleName,
                                       'Position.csv'), dtype=str, delimiter=',')
                positionRecords.extend(nodulePosition)
                for sample in nodulePosition:
                    if sample[0] not in positionDOISet:
                        positionDOISet[sample[0]] = 1
                    else:
                        positionDOISet[sample[0]] += 1

            DCMPosition = {}
            for UID in positionDOISet.keys():
                records = []
                for sample in positionRecords:
                    if sample[0] == UID:
                        records.append([float(sample[1]), float(sample[2])])
                DCMPosition[UID] = [int(numpy.median(records, axis=0)[0]), int(numpy.median(records, axis=0)[1])]

            # for sample in DCMPosition.keys():
            #     print(sample, DCMPosition[sample])

            for searchIndexA in os.listdir(os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')])):
                for searchIndexB in os.listdir(
                        os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')], searchIndexA)):
                    for dcmFile in os.listdir(
                            os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')], searchIndexA, searchIndexB)):
                        if dcmFile[-3:] != 'dcm': continue
                        DCMFile = pydicom.read_file(
                            os.path.join(LIDCPath, InstanceName[0:InstanceName.find('.')], searchIndexA, searchIndexB,
                                         dcmFile))
                        currentUID = DCMFile.SOPInstanceUID
                        if DCMFile.SOPInstanceUID in DCMPosition.keys():
                            yPosition = DCMPosition[currentUID][0]
                            xPosition = DCMPosition[currentUID][1]
                            # print(xPosition, yPosition)
                            if xPosition < FIGURE_SCOPE: xPosition = FIGURE_SCOPE
                            if xPosition > 512 - FIGURE_SCOPE: xPosition = 512 - FIGURE_SCOPE
                            if yPosition < FIGURE_SCOPE: yPosition = FIGURE_SCOPE
                            if yPosition > 512 - FIGURE_SCOPE: yPosition = 512 - FIGURE_SCOPE
                            xPosition -= FIGURE_SCOPE
                            yPosition -= FIGURE_SCOPE
                            # plt.imshow(
                            #     DCMFile.pixel_array[xPosition:xPosition + 2 * FIGURE_SCOPE,
                            #     yPosition:yPosition + 2 * FIGURE_SCOPE])
                            # plt.show()

                            fileCounter = 0
                            while True:
                                fileCounter += 1
                                if not os.path.exists(os.path.join(SavePath, InstanceName, treatNodule[0],
                                                                   'Part%04d.csv' % fileCounter)):
                                    break

                            with open(
                                    os.path.join(SavePath, InstanceName, treatNodule[0], 'Part%04d.csv' % fileCounter),
                                    'w') as file:
                                for indexX in range(2 * FIGURE_SCOPE):
                                    for indexY in range(2 * FIGURE_SCOPE):
                                        if indexY != 0: file.write(',')
                                        file.write(str(DCMFile.pixel_array[indexX + xPosition, indexY + yPosition]))
                                    file.write('\n')
                            # exit()
