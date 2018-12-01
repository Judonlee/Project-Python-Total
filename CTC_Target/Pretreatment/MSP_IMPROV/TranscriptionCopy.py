import os
from shutil import copy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/MSP-IMPROV_text/MSP-IMPROV_text/'
    datapath = 'D:/ProjectData/MSP-IMPROVE/Voice/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/Transcription/'
    for filename in os.listdir(loadpath):
        findFlag = False
        for indexA in os.listdir(datapath):
            if findFlag: break
            for indexB in os.listdir(os.path.join(datapath, indexA)):
                if findFlag: break
                for indexC in os.listdir(os.path.join(datapath, indexA, indexB)):
                    if findFlag: break
                    for indexD in os.listdir(os.path.join(datapath, indexA, indexB, indexC)):
                        # print(indexD[0:indexD.find('.')])
                        if indexD[0:indexD.find('.')] == filename[0:filename.find('.')]:
                            if not os.path.exists(os.path.join(savepath, indexA, indexB, indexC)):
                                os.makedirs(os.path.join(savepath, indexA, indexB, indexC))
                            print(indexD)
                            copy(os.path.join(loadpath, filename),
                                 os.path.join(savepath, indexA, indexB, indexC, filename))
                            findFlag = True
                            break
        # exit()
