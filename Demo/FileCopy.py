import os
import shutil

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Voices/'
    savepath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Voices-Choosed/'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    if not os.path.exists(os.path.join(savepath, indexA, indexB, indexC, indexD)):
                        os.makedirs(os.path.join(savepath, indexA, indexB, indexC, indexD))
                    for indexE in os.listdir(os.path.join(loadpath, indexA, indexB, indexC, indexD)):
                        print(indexA, indexB, indexC, indexD, indexE)
                        shutil.copy(os.path.join(loadpath, indexA, indexB, indexC, indexD, indexE),
                                    os.path.join(savepath, indexA, indexB, indexC, indexD, indexE))
