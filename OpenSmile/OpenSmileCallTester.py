from OpenSmile.OpenSmileCall import OpenSmileCall_Single
import os

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/Voice/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/OpenSmile/emobase/'
    conf = 'emobase.conf'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                if os.path.exists(os.path.join(savepath, indexA, indexB, indexC)): continue
                os.makedirs(os.path.join(savepath, indexA, indexB, indexC))

                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    if os.path.exists(os.path.join(savepath, indexA, indexB, indexC, indexD + '.csv')): continue
                    print(indexA, indexB, indexC, indexD)
                    OpenSmileCall_Single(loadpath=os.path.join(loadpath, indexA, indexB, indexC, indexD),
                                         confPath=conf,
                                         savepath=os.path.join(savepath, indexA, indexB, indexC, indexD + '.csv'))
