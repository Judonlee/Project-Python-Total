import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Transcription/'
    savepath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Transcription-Npy/'
    os.makedirs(savepath)
    for indexA in os.listdir(loadpath):
        totalScription = []
        for indexB in os.listdir(loadpath + indexA):
            for indexC in os.listdir(loadpath + indexA + '/' + indexB):
                file = open(loadpath + indexA + '/' + indexB + '/' + indexC, 'r')
                transcription = file.read()
                file.close()
                print(indexA, indexB, indexC, transcription.count(' '))
                sequence = numpy.ones(transcription.count(' '))

                if indexB == 'IDL': sequence = sequence * 0
                if indexB == 'NEG': sequence = sequence * 1

                if indexB == 'A': sequence = sequence * 0
                if indexB == 'E': sequence = sequence * 1
                if indexB == 'N': sequence = sequence * 2
                if indexB == 'P': sequence = sequence * 3
                if indexB == 'R': sequence = sequence * 4

                print(sequence)
                totalScription.append(sequence)
        print(numpy.shape(totalScription))
        numpy.save(savepath + indexA + '.npy', totalScription)
