import numpy
import os


def IEMOCAP_Spectrogram_Loader(bands, appoint=0):
    datapath = 'D:\\ProjectData\\IEMOCAP-Npy\\Bands' + str(bands) + '\\'
    trainData, trainLabel, trainSeq, testData, testLabel, testSeq = [], [], [], [], [], []

    for indexA in os.listdir(datapath):
        for indexB in os.listdir(datapath + indexA):
            for indexC in os.listdir(datapath + indexA + '\\' + indexB):
                for indexD in ['ang', 'exc', 'hap', 'neu', 'sad']:
                    data = numpy.load(datapath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '-Data.npy')
                    seq = numpy.load(datapath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '-Seq.npy')

                    label = numpy.zeros((len(data), 4))

                    if indexD == 'ang': appointLabel = 0
                    if indexD == 'exc' or indexD == 'hap': appointLabel = 1
                    if indexD == 'neu': appointLabel = 2
                    if indexD == 'sad': appointLabel = 3
                    for index in range(len(label)):
                        label[index][appointLabel] = 1

                    counter = ['Female', 'Male'].index(indexB) * 5 + \
                              ['Session1', 'Session2', 'Session3', 'Session4', 'Session5'].index(indexC)
                    if appoint == counter:
                        testData.extend(data)
                        testSeq.extend(seq)
                        testLabel.extend(label)
                    else:
                        trainData.extend(data)
                        trainSeq.extend(seq)
                        trainLabel.extend(label)
                    print('Loading :', indexA, indexB, indexC, indexD, counter)
    print('Train Part :', numpy.shape(trainData), numpy.shape(trainSeq), numpy.shape(trainLabel),
          numpy.sum(trainLabel, axis=0))
    print('Test Part :', numpy.shape(testData), numpy.shape(testSeq), numpy.shape(testLabel),
          numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, trainSeq, testData, testLabel, testSeq


if __name__ == '__main__':
    IEMOCAP_Spectrogram_Loader(bands=30)

'''
if __name__ == '__main__':

    datapath = 'D:\\ProjectData\\IEMOCAP-Normalized\\Bands120\\'
    savepath = 'D:\\ProjectData\\IEMOCAP-Npy\\Bands120\\'

    maxlen = 3500
    for indexA in os.listdir(datapath):
        for indexB in os.listdir(datapath + indexA):
            for indexC in os.listdir(datapath + indexA + '\\' + indexB):
                os.makedirs(savepath + indexA + '\\' + indexB + '\\' + indexC)
                for indexD in os.listdir(datapath + indexA + '\\' + indexB + '\\' + indexC):
                    print(indexA, indexB, indexC, indexD)
                    totalData, depth = [], []
                    for indexE in os.listdir(datapath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD):
                        currentData = numpy.genfromtxt(
                            datapath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '\\' + indexE,
                            dtype=float, delimiter=',')
                        depth.append(len(currentData))
                        currentData = numpy.concatenate(
                            (currentData, numpy.zeros((maxlen - len(currentData), len(currentData[0])))), axis=0)
                        # print(numpy.shape(currentData))
                        totalData.append(currentData)
                    numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '-Data.npy',
                               totalData)
                    numpy.save(savepath + indexA + '\\' + indexB + '\\' + indexC + '\\' + indexD + '-Label.npy',
                               depth)
'''
