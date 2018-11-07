import numpy
import os

if __name__ == '__main__':
    loadpath = 'E:/Project-CTC-Data/Csv-Normalized/Bands120/'
    savepath = 'D:/ProjectData/IEMOCAP-New-Again/Bands120/'
    transcriptionPath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU/'
    os.makedirs(savepath)

    for indexA in ['improve']:
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                print(indexA, indexB, indexC)
                totalData, totalLabel, totalSeq, totalTranscription = [], [], [], []
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    for indexE in os.listdir(os.path.join(loadpath, indexA, indexB, indexC, indexD)):
                        currentData = numpy.genfromtxt(os.path.join(loadpath, indexA, indexB, indexC, indexD, indexE),
                                                       dtype=float, delimiter=',')
                        if indexD == 'ang': currentLabel = [1, 0, 0, 0]
                        if indexD == 'exc' or indexD == 'hap': currentLabel = [0, 1, 0, 0]
                        if indexD == 'neu': currentLabel = [0, 0, 1, 0]
                        if indexD == 'sad': currentLabel = [0, 0, 0, 1]

                        file = open(os.path.join(transcriptionPath, indexA, indexB, indexC, indexD,
                                                 indexE[0:indexE.find('.')] + '.txt'), 'r')
                        data = file.read()
                        file.close()
                        # print(indexA, indexB, indexC, indexD, indexE, currentTranscription)

                        currentScription = [0]
                        for index in range(data.count(' ') + 1):
                            currentScription.append(1)
                            currentScription.append(0)
                        # print(data)
                        # print(currentScription)
                        # exit()

                        totalData.append(currentData)
                        totalLabel.append(currentLabel)
                        totalSeq.append(len(currentData))
                        totalTranscription.append(currentScription)
                print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.sum(totalLabel, axis=0),
                      numpy.shape(totalSeq), numpy.shape(totalTranscription))
                numpy.save(savepath + indexB + '-' + indexC + '-Data.npy', totalData)
                numpy.save(savepath + indexB + '-' + indexC + '-Label.npy', totalLabel)
                numpy.save(savepath + indexB + '-' + indexC + '-Seq.npy', totalSeq)
                numpy.save(savepath + indexB + '-' + indexC + '-Scription.npy', totalTranscription)
