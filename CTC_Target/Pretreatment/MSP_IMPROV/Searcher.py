import os
import numpy

if __name__ == '__main__':
    featurePath = 'D:/ProjectData/MSP-IMPROVE/Voice-Normalized/Bands-30/'
    transcriptPath = 'D:/ProjectData/MSP-IMPROVE/Transcription-CMU/'

    for indexA in os.listdir(featurePath):
        for indexB in os.listdir(os.path.join(featurePath, indexA)):
            print(indexA, indexB)
            for indexC in os.listdir(os.path.join(featurePath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(featurePath, indexA, indexB, indexC)):
                    feature = numpy.genfromtxt(os.path.join(featurePath, indexA, indexB, indexC, indexD), dtype=float,
                                               delimiter=',')
                    with open(os.path.join(transcriptPath, indexA, indexB, indexC, indexD[0:indexD.find('.')] + '.txt'),
                              'r') as file:
                        transcriptionData = file.read()

                    tranScriptionReal = numpy.ones(transcriptionData.count(' ') + 1)
                    # print(numpy.shape(feature), len(tranScriptionReal))
                    if numpy.shape(feature)[0] < len(tranScriptionReal):
                        print(indexA, indexB, indexC, indexD)
                        # exit()
                    # exit()
