import os
import numpy
import pydicom
import matplotlib.pylab as plt

FIGURE_SCOPE = 32

if __name__ == '__main__':
    LIDCPath = 'E:/LIDC/LIDC-IDRI/'
    InstancePath = 'E:/LIDC/TreatmentTrace/Step1-InstanceNumber/'
    SavePath = 'E:/LIDC/TreatmentTrace/Step5-NonNodules-MediaPosition/'
    os.makedirs(SavePath)
    for indexA in os.listdir(LIDCPath):
        if not os.path.exists(os.path.join(InstancePath, indexA + '.csv')): continue
        instanceData = numpy.genfromtxt(os.path.join(InstancePath, indexA + '.csv'), dtype=str, delimiter=',')
        # print(instanceData)
        instanceDictionary = {}
        for sample in instanceData:
            instanceDictionary[sample[1]] = sample[0]

        print(indexA)
        for indexB in os.listdir(os.path.join(LIDCPath, indexA)):
            for indexC in os.listdir(os.path.join(LIDCPath, indexA, indexB)):
                for xmlFile in os.listdir(os.path.join(LIDCPath, indexA, indexB, indexC)):
                    if xmlFile[-3:] != 'xml': continue
                    with open(os.path.join(LIDCPath, indexA, indexB, indexC, xmlFile), 'r') as file:
                        xmlData = file.read()

                    startPosition = 0
                    nonNoduleList = []
                    while xmlData.find('<nonNodule>', startPosition) != -1:
                        startPosition = xmlData.find('<nonNodule>', startPosition)
                        UID = xmlData[xmlData.find('<imageSOP_UID>', startPosition) + len('<imageSOP_UID>'):
                                      xmlData.find('</imageSOP_UID>', startPosition)]
                        yPosition = int(xmlData[xmlData.find('<xCoord>', startPosition) + len('<xCoord>'):
                                                xmlData.find('</xCoord>', startPosition)])
                        xPosition = int(xmlData[xmlData.find('<yCoord>', startPosition) + len('<yCoord>'):
                                                xmlData.find('</yCoord>', startPosition)])
                        startPosition = xmlData.find('</nonNodule>', startPosition)
                        # print(UID, xPosition, yPosition)

                        if xPosition < FIGURE_SCOPE: xPosition = FIGURE_SCOPE
                        if xPosition > 512 - FIGURE_SCOPE: xPosition = 512 - FIGURE_SCOPE
                        if yPosition < FIGURE_SCOPE: yPosition = FIGURE_SCOPE
                        if yPosition > 512 - FIGURE_SCOPE: yPosition = 512 - FIGURE_SCOPE
                        xPosition -= FIGURE_SCOPE
                        yPosition -= FIGURE_SCOPE
                        nonNoduleList.append([UID, xPosition, yPosition])

                    with open(os.path.join(SavePath, indexA + '.csv'), 'w')as file:
                        for nonNoduleSample in nonNoduleList:
                            if nonNoduleSample[0] not in instanceDictionary.keys(): continue
                            file.write(instanceDictionary[nonNoduleSample[0]] + ',' + str(nonNoduleSample[1]) +
                                       ',' + str(nonNoduleSample[2]) + '\n')
