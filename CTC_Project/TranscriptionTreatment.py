import os

if __name__ == '__main__':
    wavpath = 'D:\\ProjectData\\IEMOCAP\\'
    labelpath = 'F:\\IEMOCAP\\IEMOCAP_full_release_withoutVideos\\IEMOCAP_full_release\\Session5\\dialog\\transcriptions\\'
    savepath = 'D:\\ProjectData\\IEMOCAP-Transcription\\'

    for filename in os.listdir(labelpath):
        file = open(file=labelpath + filename, mode='r')
        data = file.readlines()
        for sample in data:
            findname = sample[0:sample.find(' ')]

            findFlag = False
            for indexA in os.listdir(wavpath):
                if findFlag: break
                for indexB in os.listdir(wavpath + indexA):
                    if findFlag: break
                    for indexC in os.listdir(wavpath + indexA + '\\' + indexB):
                        if findFlag: break
                        for indexD in os.listdir(wavpath + indexA + '\\' + indexB + '\\' + indexC):
                            if findFlag: break
                            for indexE in os.listdir(wavpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                                if indexE == findname + '.wav':
                                    print(indexA, indexB, indexC, indexD, indexE)
                                    findFlag = True

                                    if not os.path.exists(
                                            savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                                        os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)
                                    writefile = open(
                                        savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + findname + '.txt',
                                        'w')
                                    writefile.write(sample[sample.find(']: ') + 3:])
                                    writefile.close()

                                if findFlag: break
            # exit()
        file.close()
