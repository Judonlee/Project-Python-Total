import numpy
import os

if __name__ == '__main__':
    loadpath = 'E:/LIDC/LIDC-IDRI/'
    savepath = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition'
    for indexA in os.listdir(loadpath):
        if os.path.exists(os.path.join(savepath, indexA)): continue
        os.makedirs(os.path.join(savepath, indexA))

        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    if indexD[-3:] != 'xml': continue
                    print(indexA, indexD)

                    with open(os.path.join(loadpath, indexA, indexB, indexC, indexD), 'r') as xmlFile:
                        xmlData = xmlFile.read()

                    startPosition = 0
                    while xmlData.find('<characteristics>', startPosition) != -1:
                        startPosition = xmlData.find('<characteristics>', startPosition)
                        endPosition = xmlData.find('</unblindedReadNodule>', startPosition)

                        noduleCounter = 0
                        while True:
                            noduleCounter += 1
                            if os.path.exists(os.path.join(savepath, indexA, 'Nodule-%04d' % noduleCounter)): continue
                            os.makedirs(os.path.join(savepath, indexA, 'Nodule-%04d' % noduleCounter))
                            break

                        with open(os.path.join(savepath, indexA, 'Nodule-%04d' % noduleCounter, 'Character.txt'),
                                  'w') as file:
                            file.write(xmlData[startPosition:xmlData.find('</characteristics>', startPosition) + len(
                                '</characteristics>')])

                        with open(os.path.join(savepath, indexA, 'Nodule-%04d' % noduleCounter, 'Position.csv'),
                                  'w') as file:
                            roiStartPosition = xmlData.find('<roi>', startPosition)
                            while xmlData.find('</roi>', roiStartPosition) != -1 and \
                                    xmlData.find('</roi>', roiStartPosition) < endPosition:
                                SOP_UID = xmlData[
                                          xmlData.find('<imageSOP_UID>', roiStartPosition) + len('<imageSOP_UID>'):
                                          xmlData.find('</imageSOP_UID>', roiStartPosition)]

                                while xmlData.find('<xCoord>', roiStartPosition) != -1 and \
                                        xmlData.find('<xCoord>', roiStartPosition) < \
                                        xmlData.find('</roi>', roiStartPosition):
                                    roiStartPosition = xmlData.find('<xCoord>', roiStartPosition)
                                    xPosition = xmlData[xmlData.find('<xCoord>', roiStartPosition) + len('<xCoord>'):
                                                        xmlData.find('</xCoord>', roiStartPosition)]
                                    roiStartPosition = xmlData.find('<yCoord>', roiStartPosition)
                                    yPosition = xmlData[xmlData.find('<yCoord>', roiStartPosition) + len('<yCoord>'):
                                                        xmlData.find('</yCoord>', roiStartPosition)]
                                    file.write(SOP_UID + ',' + xPosition + ',' + yPosition + '\n')

                                roiStartPosition = xmlData.find('</roi>', roiStartPosition) + 1

                        startPosition = endPosition
                    # exit()
