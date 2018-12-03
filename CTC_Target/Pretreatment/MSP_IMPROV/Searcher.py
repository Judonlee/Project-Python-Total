import os
import numpy

if __name__ == '__main__':
    voicepath = 'D:/ProjectData/MSP-IMPROVE/Voice-Target/read/'
    transcriptionpath = 'D:/ProjectData/MSP-IMPROVE/Transcription/'

    matrix = numpy.zeros((6, 4))
    for indexA in os.listdir(voicepath):
        for indexB in os.listdir(os.path.join(voicepath, indexA)):
            for indexC in os.listdir(os.path.join(voicepath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(voicepath, indexA, indexB, indexC)):
                    if not os.path.exists(os.path.join(transcriptionpath, indexA, indexB, indexC,
                                                       indexD[0:indexD.find('.')] + '.txt')):
                        print(indexA, indexB, indexC, indexD)
                        matrix[int(indexA[-1]) - 1][['A', 'H', 'N', 'S'].index(indexC)] += 1
                        # counter += 1
                    # print(indexA, indexB, indexC)
    # print(counter)
    for indexX in range(numpy.shape(matrix)[0]):
        for indexY in range(numpy.shape(matrix)[1]):
            print(matrix[indexX][indexY], end='\t')
        print()
