import os


def OpenSmileCall_Single(loadpath, confPath, savepath):
    commandPath = r'D:\OpenSmile\opensmile-2.3.0-Single\bin\Win32\SMILExtract_Release'
    confPath = 'D:\\OpenSmile\\opensmile-2.3.0-Single\\config\\' + confPath
    os.system(commandPath + ' -C ' + confPath + ' -I ' + loadpath + ' -O ' + savepath + '.current')

    loadfile = open(savepath + '.current', 'r')
    data = loadfile.readlines()
    loadfile.close()

    file = open(savepath, 'w')
    for sample in data:
        if sample[0] == '@': continue
        if len(sample) < 5: continue
        file.write(sample[sample.find(',') + 1:-3] + '\n')
    file.close()
    os.remove(savepath + '.current')


if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\IEMOCAP\\IEMOCAP-Voices\\'

    confList = ['IS09']
    confPath = ['IS09_emotion.conf']

    for confIndex in range(1):
        savepath = 'D:\\ProjectData\\IEMOCAP\\IEMOCAP-Voices-Features\\' + confList[confIndex] + '\\'
        for indexA in ['improve', 'script']:
            for indexB in os.listdir(loadpath + indexA):
                for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                    for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                        if os.path.exists(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD): continue
                        os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)

                        for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                            print(confList[confIndex], indexA, indexB, indexC, indexD, indexE)
                            if os.path.exists(
                                    savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE + '.csv'): continue
                            OpenSmileCall_Single(
                                loadpath=loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                confPath=confPath[confIndex],
                                savepath=savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE + '.csv')
                            # exit()
