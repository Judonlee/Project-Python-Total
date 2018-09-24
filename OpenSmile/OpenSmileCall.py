import os


def OpenSmileCall_Sequence(loadpath, confPath, savepath):
    commandPath = r'D:\opensmile-2.3.0\bin\Win32\SMILExtract_Release'
    confPath = 'D:\\opensmile-2.3.0\\config\\' + confPath
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
    loadpath = 'D:\\ProjectData\\IEMOCAP\\'
    savepath = 'D:\\ProjectData\\IEMOCAP-Features\\GeMAPS\\'
    for indexA in os.listdir(loadpath)[1:]:
        for indexB in os.listdir(loadpath + indexA)[0:1]:
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                for indexD in ['ang', 'hap', 'exc', 'neu', 'sad']:
                    os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        print(indexA, indexB, indexC, indexD, indexE)
                        OpenSmileCall_Sequence(
                            loadpath=loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            confPath='gemaps\\GeMAPSv01a.conf',
                            savepath=savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE + '.csv')
