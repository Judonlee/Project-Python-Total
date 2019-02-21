import os
import numpy

if __name__ == '__main__':
    loadpath = 'D:/LIDC/LBP/R=3_P=24_Csv/'
    savepath = 'D:/LIDC/LBP-Npy/R=3_P=24/'

    if not os.path.exists(savepath): os.makedirs(savepath)
    for part in range(5):
        totalData, totalLabel = [], []
        for indexA in os.listdir(os.path.join(loadpath, 'Part%d' % part, 'Nodules')):
            print(part, 'Nodules', indexA)
            for indexB in os.listdir(os.path.join(loadpath, 'Part%d' % part, 'Nodules', indexA)):
                for indexC in os.listdir(os.path.join(loadpath, 'Part%d' % part, 'Nodules', indexA, indexB)):
                    data = numpy.genfromtxt(
                        fname=os.path.join(loadpath, 'Part%d' % part, 'Nodules', indexA, indexB, indexC), dtype=int,
                        delimiter=',')
                    totalData.append(data)
                    totalLabel.append([1, 0])
        for indexA in os.listdir(os.path.join(loadpath, 'Part%d' % part, 'NonNodules')):
            print(part, 'NonNodules', indexA)
            for indexB in os.listdir(os.path.join(loadpath, 'Part%d' % part, 'NonNodules', indexA)):
                data = numpy.genfromtxt(fname=os.path.join(loadpath, 'Part%d' % part, 'NonNodules', indexA, indexB),
                                        dtype=int, delimiter=',')
                totalData.append(data)
                totalLabel.append([0, 1])
        print(numpy.shape(totalData), numpy.shape(totalLabel), numpy.sum(totalLabel, axis=0))
        numpy.save(os.path.join(savepath, 'Part%d-Data.npy' % part), totalData)
        numpy.save(os.path.join(savepath, 'Part%d-Label.npy' % part), totalLabel)
