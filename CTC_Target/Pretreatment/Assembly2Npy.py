import numpy
import os

if __name__ == '__main__':
    loadPath = 'D:/ProjectData/FAU-AEC-Treated/IS2009-Class5-Csv-Normalized/Bands-40/Mont/'
    transcriptionPath = 'D:/ProjectData/FAU-AEC-Treated/Transcription/'
    totalData, totalLabel, totalTranscription, totalSeq = [], [], [], []
    for indexA in os.listdir(loadPath):
        for indexB in os.listdir(os.path.join(loadPath, indexA)):
            print(indexA, indexB)

            data = numpy.genfromtxt(fname=os.path.join(loadPath, indexA, indexB), dtype=float, delimiter=',')
            seqLen = len(data)
            with open(os.path.join(transcriptionPath, indexB[0:indexB.find('.')] + '.txt'), 'r')as file:
                trans = file.read()
                transcription = numpy.ones(trans.count(' ') + 1)

            if indexA == 'A': transcription *= 0;label = [1, 0, 0, 0, 0]
            if indexA == 'E': transcription *= 1;label = [0, 1, 0, 0, 0]
            if indexA == 'N': transcription *= 2;label = [0, 0, 1, 0, 0]
            if indexA == 'P': transcription *= 3;label = [0, 0, 0, 1, 0]
            if indexA == 'R': transcription *= 4;label = [0, 0, 0, 0, 1]
            # print(transcription)

            totalData.append(data)
            totalLabel.append(label)
            totalTranscription.append(transcription)
            totalSeq.append(seqLen)
    print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.shape(totalTranscription), numpy.shape(totalSeq),
          numpy.sum(totalLabel, axis=0))
    numpy.save('Mont-Data.npy', totalData)
    numpy.save('Mont-Label.npy', totalLabel)
    numpy.save('Mont-Transcription.npy', totalTranscription)
    numpy.save('Mont-Seq.npy', totalSeq)
