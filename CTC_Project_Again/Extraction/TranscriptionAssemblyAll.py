import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription/'
    savepath = 'D:/ProjectData/IEMOCAP/IEMOCAP-Transcription-CMU/'

    dictionary = numpy.genfromtxt('D:/GitHub/CTC_Project_Again/Extraction/Dictionary.txt', dtype=str, delimiter='  ')

    dict = {}
    for sample in dictionary:
        dict[sample[0]] = sample[1]

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    os.makedirs(os.path.join(savepath, indexA, indexB, indexC, indexD))
                    for indexE in os.listdir(os.path.join(loadpath, indexA, indexB, indexC, indexD)):
                        print(indexA, indexB, indexC, indexD, indexE)
                        currentFile = open(os.path.join(loadpath, indexA, indexB, indexC, indexD, indexE), 'r')
                        data = currentFile.read()
                        currentFile.close()
                        data = data.replace('.', '')
                        data = data.replace(',', '')
                        data = data.replace('?', '')
                        data = data.replace('-', '')
                        data = data.replace('\n', '')

                        file = open(os.path.join(savepath, indexA, indexB, indexC, indexD, indexE), 'w')
                        for sample in data.upper().split(' '):
                            if sample in dict.keys():
                                file.write(dict[sample])
                                file.write(' ')
                        file.close()
