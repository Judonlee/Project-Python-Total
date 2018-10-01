import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\IEMOCAP-Transcription\\'
    savepath = 'D:\\ProjectData\\Project-CTC-Data\\Transcription-IntersectionWordNumber-Class6\\'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            os.makedirs(savepath + indexA + '\\' + indexB)
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                print(indexA, indexB, indexC)
                transcription = []

                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        file = open(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                    'r')
                        data = file.read()
                        file.close()
                        print(data.count(' '))

                        if indexD == 'ang': appoint = 1
                        if indexD == 'exc' or indexD == 'hap': appoint = 2
                        if indexD == 'neu': appoint = 3
                        if indexD == 'sad': appoint = 4

                        currentTranscription = [appoint]
                        for indexX in range(data.count(' ')):
                            currentTranscription.append(0)
                            currentTranscription.append(appoint)
                        transcription.append(currentTranscription)
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '.npy', transcription)
