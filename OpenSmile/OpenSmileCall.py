import os


def OpenSmileCall_Sequence(loadpath, confPath, savepath):
    commandPath = r'D:\OpenSmile\opensmile-2.3.0-Sequence\bin\Win32\SMILExtract_Release'
    confPath = 'D:\\OpenSmile\\opensmile-2.3.0-Sequence\\config\\' + confPath
    os.system(commandPath + ' -C ' + confPath + ' -I ' + loadpath + ' -O ' + savepath + '.csv')
    '''
    loadfile = open(savepath + '.current', 'r')
    data = loadfile.readlines()
    loadfile.close()

    file = open(savepath, 'w')
    for sample in data:
        if sample[0] == '@': continue
        if len(sample) < 5: continue
        file.write(sample[sample.find(',') + 1:-3] + '\n')
    file.close()
    os.remove(savepath + '.current')'''


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

    confList = ['avec2011', 'avec2013', 'ComParE', 'emobase', 'IS09', 'IS10', 'IS11', 'IS12', 'IS13']
    confPath = ['avec2011.conf', 'avec2013.conf', 'ComParE_2016.conf', 'emobase.conf', 'IS09_emotion.conf',
                'IS10_paraling.conf', 'IS11_speaker_state.conf', 'IS12_speaker_trait.conf', 'IS13_ComParE.conf']
    for confIndex in range(8, len(confList)):
        savepath = 'D:\\ProjectData\\IEMOCAP\\IEMOCAP-Features\\' + confList[confIndex] + '-Sequence\\'
        for indexA in os.listdir(loadpath):
            for indexB in os.listdir(loadpath + indexA):
                for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                    for indexD in ['ang', 'hap', 'exc', 'neu', 'sad']:
                        if not os.path.exists(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                            os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD)
                        for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                            print(confList[confIndex], indexA, indexB, indexC, indexD, indexE)
                            if os.path.exists(
                                    savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE + '.csv'): continue
                            OpenSmileCall_Sequence(
                                loadpath=loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                confPath=confPath[confIndex],
                                savepath=savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE + '.csv')
                            # exit()
