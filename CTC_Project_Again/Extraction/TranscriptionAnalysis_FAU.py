import os

if __name__ == '__main__':
    loadpath = 'F:/transliteration/transliteration.txt'
    classifiedpath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5/'
    savepath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Transcription/'
    file = open(loadpath, 'r')
    data = file.readlines()
    file.close()

    for sample in data:
        # print(sample)
        filename = sample[0:sample.find(' ')]
        print(filename)
        flag = False
        for indexA in os.listdir(classifiedpath):
            if flag: break
            for indexB in os.listdir(classifiedpath + indexA):
                if flag: break
                for wavname in os.listdir(classifiedpath + indexA + '/' + indexB):
                    if flag: break
                    if filename == wavname[0:wavname.find('.')]:
                        flag = True
                        if not os.path.exists(savepath + indexA + '/' + indexB):
                            os.makedirs(savepath + indexA + '/' + indexB)
                        file = open(savepath + indexA + '/' + indexB + '/' + filename + '.txt', 'w')
                        file.write(sample[sample.find(' ') + 1:])
                        file.close()
