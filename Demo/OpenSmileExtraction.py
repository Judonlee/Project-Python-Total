import os


def OpenSmileCall_Single(loadpath, confPath, savepath):
    commandPath = r'D:\OpenSmile\opensmile-2.3.0-Single\bin\Win32\SMILExtract_Release'
    confPath = 'D:\\OpenSmile\\opensmile-2.3.0-Single\\config\\' + confPath
    print(commandPath + ' -C ' + confPath + ' -I ' + loadpath + ' -O ' + savepath + '.current')
    # return

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
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Voices-Choosed/'
    savepath = 'D:/ProjectData/IEMOCAP/IEMOCAP-OpenSmile/IS09/'
    confName = 'IS09_emotion.conf'

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    if not os.path.exists(os.path.join(savepath, indexA, indexB, indexC, indexD)):
                        os.makedirs(os.path.join(savepath, indexA, indexB, indexC, indexD))

                    for indexE in os.listdir(os.path.join(loadpath, indexA, indexB, indexC, indexD)):
                        OpenSmileCall_Single(
                            loadpath=os.path.join(loadpath, indexA, indexB, indexC, indexD, indexE),
                            confPath=confName,
                            savepath=os.path.join(savepath, indexA, indexB, indexC, indexD, indexE + '.csv'))
                        #exit()
