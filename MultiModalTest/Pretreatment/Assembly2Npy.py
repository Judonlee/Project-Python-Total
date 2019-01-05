import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/Features/IEMOCAP-Normalized/Bands40/improve/'
    labelpath = 'D:/ProjectData/Features/IEMOCAP-Labels/CMU-Label/improve/'
    savepath = 'D:/ProjectData/Features/IEMOCAP-Npy/Bands40/'

    os.makedirs(savepath)

    dictionary = {"AA": 0, "AE": 1, "AH": 2, "AO": 3, "AW": 4, "AY": 5, "B": 6, "CH": 7, "D": 8, "DH": 9, "EH": 10,
                  "ER": 11, "EY": 12, "F": 13, "G": 14, "HH": 15, "IH": 16, "IY": 17, "JH": 18, "K": 19, "L": 20,
                  "M": 21, "N": 22, "NG": 23, "OW": 24, "OY": 25, "P": 26, "R": 27, "S": 28, "SH": 29, "T": 30,
                  "TH": 31, "UH": 32, "UW": 33, "V": 34, "W": 35, "Y": 36, "Z": 37, "ZH": 38}

    # totalTranscription = []
    for indexA in os.listdir(labelpath):
        for indexB in os.listdir(os.path.join(labelpath, indexA)):

            partData, partLabel = [], []

            for indexC in os.listdir(os.path.join(labelpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(labelpath, indexA, indexB, indexC)):
                    print(indexA, indexB, indexC, indexD)
                    with open(os.path.join(labelpath, indexA, indexB, indexC, indexD), 'r') as file:
                        data = file.read()
                        data = data.split(' ')[0:-1]
                        if len(data) == 0:
                            data = ['AA']

                        print(data)

                        singleTranscription = []

                        for sample in data:
                            compare = sample
                            if (compare[-1] >= '0') and (compare[-1] <= '9'):
                                compare = compare[0:-1]
                            singleTranscription.append(dictionary[compare])
                        print(singleTranscription)

                        partLabel.append(singleTranscription)

                    features = numpy.genfromtxt(
                        os.path.join(loadpath, indexA, indexB, indexC, indexD[0:indexD.find('.')] + '.wav.csv'),
                        dtype=float, delimiter=',')
                    partData.append(features)
                    # exit()

            numpy.save(os.path.join(savepath, '%s-%s-Data.npy' % (indexA, indexB)), partData)
            numpy.save(os.path.join(savepath, '%s-%s-Label.npy' % (indexA, indexB)), partLabel)
