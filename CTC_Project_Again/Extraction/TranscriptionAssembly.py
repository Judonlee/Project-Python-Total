import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:\\ProjectData\\IEMOCAP-Transcription\\'
    savepath = 'F:\\Project-CTC-Data\\Transcription-SingleNumber\\'
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(loadpath + indexA):
            if not os.path.exists(savepath + indexA + '\\' + indexB):
                os.makedirs(savepath + indexA + '\\' + indexB)
            for indexC in os.listdir(loadpath + indexA + '\\' + indexB):
                totalTranscription = []
                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    for indexE in os.listdir(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        file = open(loadpath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                                    'r')
                        data = file.read()
                        file.close()
                        transcription = numpy.ones(data.count(' ') + 1)
                        if indexD == 'ang': transcription = transcription * 1
                        if indexD == 'exc' or indexD == 'hap': transcription = transcription * 2
                        if indexD == 'neu': transcription = transcription * 3
                        if indexD == 'sad': transcription = transcription * 4
                        totalTranscription.append(transcription)
                print(indexA, indexB, indexC, numpy.shape(totalTranscription))
                numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '.npy', totalTranscription)
