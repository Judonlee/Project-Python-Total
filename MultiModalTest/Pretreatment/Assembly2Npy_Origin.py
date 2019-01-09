import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/Features/IEMOCAP-Normalized/Bands30/improve/'
    savepath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands30-CNN/'

    os.makedirs(savepath)

    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            partData, partLabel = [], []
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    # print(indexA, indexB, indexC, indexD)

                    features = numpy.genfromtxt(
                        os.path.join(loadpath, indexA, indexB, indexC, indexD[0:indexD.find('.')] + '.wav.csv'),
                        dtype=float, delimiter=',')
                    partData.append(features)

                    if indexC == 'ang': label = [1, 0, 0, 0]
                    if indexC == 'exc' or indexC == 'hap': label = [0, 1, 0, 0]
                    if indexC == 'neu': label = [0, 0, 1, 0]
                    if indexC == 'sad': label = [0, 0, 0, 1]
                    partLabel.append(label)
            print(numpy.shape(partData), numpy.shape(partLabel))

            numpy.save(os.path.join(savepath, '%s-%s-Data.npy' % (indexA, indexB)), partData)
            numpy.save(os.path.join(savepath, '%s-%s-Label.npy' % (indexA, indexB)), partLabel)
