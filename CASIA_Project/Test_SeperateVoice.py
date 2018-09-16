import os
from pydub import AudioSegment

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\DataAugment\\'
    counter = 0
    winstep = 1
    for winlen in [8]:
        savepath = 'F:\\ProjectData\\DataAugment-Seconds' + str(winlen) + 'S\\'
        for indexA in os.listdir(loadpath):
            for indexB in os.listdir(loadpath + indexA):
                for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                    for indexD in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC):
                        for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                            # os.makedirs(
                            #    savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE)
                            print(indexA, indexB, indexC, indexD, indexE)

                            wav = AudioSegment.from_wav(
                                loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE)
                            startPosition = 0

                            wav = wav + AudioSegment.silent((int(len(wav) / 1000) + 1) * 1000 - len(wav))
                            while len(wav) <= winlen * 1000:
                                wav = wav + AudioSegment.silent(1000)

                            while startPosition + winlen <= len(wav) / 1000:
                                name = str(int(startPosition / winstep))
                                while len(name) < 4:
                                    name = '0' + name
                                counter += 1
                                '''
                                wav[startPosition * 1000:(startPosition + winlen) * 1000].export(
                                    savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE +
                                    '\\Part' + name + '.wav',
                                    format='wav')'''
                                startPosition += winstep
    print(counter)
