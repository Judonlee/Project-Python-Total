import numpy
import os

if __name__ == '__main__':
    datapath = 'D:/ProjectData/IEMOCAP/MFCC-Normalized/improve/'
    transcriptPath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription/improve/'
    savepath = 'E:/CTC_Target/Features/WordNumber/'
    os.makedirs(savepath)
    for indexA in os.listdir(datapath):
        for indexB in os.listdir(os.path.join(datapath, indexA)):
            partData, partSeq, partLabel, partTranscription = [], [], [], []
            print(indexA, indexB)
            for indexC in os.listdir(os.path.join(datapath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(datapath, indexA, indexB, indexC)):
                    data = numpy.genfromtxt(fname=os.path.join(datapath, indexA, indexB, indexC, indexD), dtype=float,
                                            delimiter=',')
                    label = numpy.zeros(4)
                    with open(os.path.join(transcriptPath, indexA, indexB, indexC, indexD[0:indexD.find('.')] + '.txt'),
                              'r') as file:
                        transcriptionData = file.read()

                    tranScriptionReal = numpy.ones(transcriptionData.count(' ') + 1)

                    if indexC == 'ang':
                        tranScriptionReal *= 0
                        label[0] = 1
                    if indexC == 'exc' or indexC == 'hap':
                        tranScriptionReal *= 1
                        label[1] = 1
                    if indexC == 'neu':
                        tranScriptionReal *= 2
                        label[2] = 1
                    if indexC == 'sad':
                        tranScriptionReal *= 3
                        label[3] = 1

                    partData.append(data)
                    partSeq.append(len(data))
                    partLabel.append(label)
                    partTranscription.append(tranScriptionReal)

            print(numpy.shape(partData), numpy.shape(partSeq), numpy.shape(partLabel), numpy.shape(partTranscription),
                  numpy.sum(partLabel, axis=0))
            # numpy.save(savepath + '%s-%s-Data.npy' % (indexA, indexB), partData)
            # numpy.save(savepath + '%s-%s-Seq.npy' % (indexA, indexB), partSeq)
            # numpy.save(savepath + '%s-%s-Label.npy' % (indexA, indexB), partLabel)
            numpy.save(savepath + '%s-%s-Transcription.npy' % (indexA, indexB), partTranscription)
