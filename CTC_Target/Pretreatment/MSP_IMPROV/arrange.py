import numpy
import os
from shutil import copy

if __name__ == '__main__':
    transcriptionPath = 'D:/ProjectData/MSP-IMPROVE/Result-Txt/'
    savePath = 'D:/ProjectData/MSP-IMPROVE/Result-Txt-Rearrange/'
    voicePath = 'D:/ProjectData/MSP-IMPROVE/Voice/'

    for indexA in os.listdir(transcriptionPath):
        for filename in os.listdir(os.path.join(transcriptionPath, indexA)):
            print(indexA, filename)

            findFlag = False
            for indexB in os.listdir(os.path.join(voicePath, indexA)):
                if findFlag: break
                for indexC in os.listdir(os.path.join(voicePath, indexA, indexB)):
                    if findFlag: break
                    for indexD in os.listdir(os.path.join(voicePath, indexA, indexB, indexC)):
                        if indexD[0:indexD.find('.')] == filename[0:filename.find('.')]:
                            # print(indexD)
                            findFlag = True
                            if not os.path.exists(os.path.join(savePath, indexA, indexB, indexC)):
                                os.makedirs(os.path.join(savePath, indexA, indexB, indexC))
                            copy(os.path.join(transcriptionPath, indexA, filename),
                                 os.path.join(savePath, indexA, indexB, indexC, filename))
                            break
            # exit()
