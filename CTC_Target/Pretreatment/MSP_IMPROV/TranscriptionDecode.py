import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/Voice-Resample-Result/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/Voice-Resample-Result-Txt/'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                os.makedirs(os.path.join(savepath, indexA, indexB, indexC))
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    print(indexA, indexB, indexC, indexD)
                    data = numpy.load(os.path.join(loadpath, indexA, indexB, indexC, indexD))
                    with open('Current.txt', 'w') as file:
                        file.write(str(data))
                    with open('Current.txt', 'r') as file:
                        data = file.read()

                    with open(os.path.join(savepath, indexA, indexB, indexC, indexD[0:indexD.find('.')] + '.txt'),
                              'w') as file:
                        startIndex = 0
                        while data.find('\'Text\'', startIndex) != -1:
                            startIndex = data.find('\'Text\'', startIndex)
                            startIndex = data.find(': ', startIndex)
                            if startIndex == -1: break
                            startIndex += len(': \"')
                            texture = data[startIndex:data.find(', ', startIndex) - 1]
                            file.write(texture + ' ')
