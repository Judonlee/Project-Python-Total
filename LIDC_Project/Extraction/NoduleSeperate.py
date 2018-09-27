import os
import pydicom
import numpy
import matplotlib.pylab as plt

if __name__ == '__main__':
    loadpath = 'F:\\LIDC\\LIDC-IDRI\\'
    savepath = 'F:\\LIDC\\LIDC-Nodules\\'
    for indexA in os.listdir(loadpath)[800:]:
        if os.path.exists(savepath + indexA): continue
        os.makedirs(savepath + indexA)

        indexB = os.listdir(loadpath + indexA)[0]
        indexC = os.listdir(loadpath + indexA + '\\' + indexB)[0]
        noduleCounter = 0

        print('Treating :', indexA)

        nowLoadPath = loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\'
        for xmlname in os.listdir(nowLoadPath):
            if xmlname[-3:] != 'xml': continue
            print(xmlname)

            xmlfile = open(nowLoadPath + xmlname, 'r')
            xmldata = xmlfile.read()
            xmlfile.close()

            indexPosition = 0
            while xmldata.find('<characteristics>', indexPosition) != -1:
                indexPosition = xmldata.find('<characteristics>', indexPosition)
                finalPosition = xmldata.find('</unblindedReadNodule>', indexPosition)

                noduleCounter += 1
                os.makedirs(savepath + indexA + '\\Nodule-%04d' % noduleCounter + '\\Csv')
                os.makedirs(savepath + indexA + '\\Nodule-%04d' % noduleCounter + '\\Png')
                file = open(savepath + indexA + '\\Nodule-%04d' % noduleCounter + '\\Character.txt', 'w')
                file.write(xmldata[indexPosition:xmldata.find('</characteristics>', indexPosition)])
                file.close()
                partCounter = 1

                while xmldata.find('<imageSOP_UID>', indexPosition) != -1 and \
                        xmldata.find('<imageSOP_UID>', indexPosition) < finalPosition:
                    indexPosition = xmldata.find('<imageSOP_UID>', indexPosition)
                    UID = xmldata[indexPosition + len('<imageSOP_UID>'):xmldata.find('</imageSOP_UID>', indexPosition)]

                    for dcmName in os.listdir(nowLoadPath):
                        if dcmName[-3:] != 'dcm': continue

                    indexPosition = xmldata.find('</imageSOP_UID>', indexPosition) + 1

                    xList, yList = [], []
                    xyPosition = indexPosition
                    while xyPosition < xmldata.find('<imageZposition>', indexPosition):
                        xyPosition = xmldata.find('<xCoord>', xyPosition)
                        xList.append(float(xmldata[xyPosition + 8:xmldata.find('</xCoord>', xyPosition)]))
                        xyPosition = xmldata.find('<yCoord>', xyPosition)
                        yList.append(float(xmldata[xyPosition + 8:xmldata.find('</yCoord>', xyPosition)]))
                    indexPosition = xmldata.find('</imageSOP_UID>', indexPosition)

                    if len(xList) == 0 or len(yList) == 0: continue

                    ###############################################################################
                    instanceFind = False
                    for DCMName in os.listdir(nowLoadPath):
                        if DCMName[-3:] != 'dcm': continue
                        try:
                            DCMFile = pydicom.read_file(nowLoadPath + DCMName)
                        except:
                            file = open(savepath + indexA + ' ' + DCMName + ' Cannot Read', 'w')
                            file.close()
                            continue

                        if DCMFile.SOPInstanceUID == UID:
                            instanceFind = True
                            print(indexA, DCMName)

                            xStart = int(numpy.median(xList)) - 40
                            yStart = int(numpy.median(yList)) - 40
                            if xStart < 0: xStart = 0
                            if yStart < 0: yStart = 0
                            if xStart > 432: xStart = 432
                            if yStart > 432: yStart = 432

                            #########################################################
                            # 存储图片
                            #########################################################

                            fig = plt.figure(figsize=(5.12, 5.12))
                            ax = fig.add_subplot(111)
                            plt.axis('off')
                            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
                            # plt.margins(0, 0)
                            try:
                                plt.imshow(DCMFile.pixel_array).set_cmap('gray')
                            except:
                                file = open(savepath + indexA + ' ' + DCMName + ' Cannot Read', 'w')
                                file.close()
                                continue

                            fig = plt.figure(figsize=(0.64, 0.64))
                            plt.axis('off')
                            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)

                            plt.imshow(DCMFile.pixel_array[yStart: yStart + 64, xStart:xStart + 64]).set_cmap('gray')
                            plt.savefig(
                                savepath + indexA + '\\Nodule-%04d' % noduleCounter + '\\Png\\Part%04d.png' % partCounter)
                            plt.clf()
                            plt.close()

                            file = open(
                                savepath + indexA + '\\Nodule-%04d' % noduleCounter + '\\Csv\\Part%04d.csv' % partCounter,
                                'w')
                            for indexY in range(80):
                                for indexX in range(80):
                                    if indexX != 0:
                                        file.write(',')
                                    file.write(str(DCMFile.pixel_array[indexY + yStart, indexX + xStart]))
                                file.write('\n')
                            file.close()

                            partCounter += 1
                            break

                indexPosition = finalPosition
        # exit()
