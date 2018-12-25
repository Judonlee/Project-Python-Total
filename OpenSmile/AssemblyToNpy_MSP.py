import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/ProjectData/MSP-IMPROVE/OpenSmile/IS12-Normalization/'
    savepath = 'D:/ProjectData/MSP-IMPROVE/OpenSmile/IS12-Npy/'
    if not os.path.exists(savepath): os.makedirs(savepath)
    for indexA in os.listdir(loadpath):
        for indexB in os.listdir(os.path.join(loadpath, indexA)):
            data, label = [], []
            for indexC in os.listdir(os.path.join(loadpath, indexA, indexB)):
                for indexD in os.listdir(os.path.join(loadpath, indexA, indexB, indexC)):
                    print(indexA, indexB, indexC, indexD)
                    currentData = numpy.genfromtxt(fname=os.path.join(loadpath, indexA, indexB, indexC, indexD),
                                                   dtype=float, delimiter=',')
                    if indexC == 'A': currentLabel = [1, 0, 0, 0]
                    if indexC == 'H': currentLabel = [0, 1, 0, 0]
                    if indexC == 'N': currentLabel = [0, 0, 1, 0]
                    if indexC == 'S': currentLabel = [0, 0, 0, 1]
                    data.append(currentData)
                    label.append(currentLabel)
            print(numpy.shape(data), numpy.shape(label), numpy.sum(label, axis=0))
            numpy.save(os.path.join(savepath, '%s-%s-Data.npy' % (indexA, indexB)), data)
            numpy.save(os.path.join(savepath, '%s-%s-Label.npy' % (indexA, indexB)), label)
