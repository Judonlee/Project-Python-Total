import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU/'
    savepath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Tran-CMU-Npy/'
    # os.makedirs(savepath)
    for indexA in ['improve']:
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                print(indexA, indexB, indexC)
                totalScription = []
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    for indexE in os.listdir(os.path.join(loadpath, indexA, indexB, indexC, indexD)):
                        file = open(os.path.join(loadpath, indexA, indexB, indexC, indexD, indexE), 'r')
                        data = file.read()
                        file.close()
                        # print(data)

                        currentScription = numpy.ones(data.count(' ') + 1)
                        if indexD == 'ang': currentScription = currentScription * 0
                        if indexD == 'exc' or indexD == 'hap': currentScription = currentScription * 1
                        if indexD == 'neu': currentScription = currentScription * 2
                        if indexD == 'sad': currentScription = currentScription * 3
                        # print(indexA, indexB, indexC, indexD, indexE, currentScription)
                        totalScription.append(currentScription)
                numpy.save(savepath + '%s-%s-Transcription.npy' % (indexB, indexC), totalScription)
