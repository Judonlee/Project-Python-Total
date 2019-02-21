import os
import numpy
import pydicom
import matplotlib.pylab as plt

FIGURE_SCOPE = 32

if __name__ == '__main__':
    LIDCPath = 'E:/LIDC/LIDC-IDRI/'
    SavePath = 'E:/LIDC/TreatmentTrace/Step5-NonNodulesCsv/'
    for indexA in os.listdir(LIDCPath):
        if os.path.exists(os.path.join(SavePath, indexA)): continue
        os.makedirs(os.path.join(SavePath, indexA))
        print(indexA)
        for indexB in os.listdir(os.path.join(LIDCPath, indexA)):
            for indexC in os.listdir(os.path.join(LIDCPath, indexA, indexB)):
                DCMDict = {}
                for dcmFile in os.listdir(os.path.join(LIDCPath, indexA, indexB, indexC)):
                    if dcmFile[-3:] != 'dcm': continue
                    DCMFile = pydicom.read_file(os.path.join(LIDCPath, indexA, indexB, indexC, dcmFile))
                    DCMDict[DCMFile.SOPInstanceUID] = dcmFile

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

                    for nonNoduleSample in nonNoduleList:
                        if nonNoduleSample[0] not in DCMDict.keys(): continue
                        DCMFile = pydicom.read_file(
                            os.path.join(LIDCPath, indexA, indexB, indexC, DCMDict[nonNoduleSample[0]]))
                        # plt.imshow(DCMFile.pixel_array[nonNoduleSample[1]:nonNoduleSample[1] + 32,
                        #            nonNoduleSample[2]:nonNoduleSample[2] + 32])
                        # plt.show()

                        partName = 1
                        while os.path.exists(os.path.join(SavePath, indexA, 'Part%04d.csv' % partName)):
                            partName += 1

                        with open(os.path.join(SavePath, indexA, 'Part%04d.csv' % partName), 'w') as file:
                            for indexX in range(2 * FIGURE_SCOPE):
                                for indexY in range(2 * FIGURE_SCOPE):
                                    if indexY != 0: file.write(',')
                                    file.write(str(DCMFile.pixel_array[nonNoduleSample[1] + indexX,
                                                                       nonNoduleSample[2] + indexY]))
                                file.write('\n')

                        # print('Got It', nonNoduleSample)

                # exit()
