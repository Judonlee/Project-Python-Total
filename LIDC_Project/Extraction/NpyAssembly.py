import numpy
import os

if __name__ == '__main__':
    nodulePath = 'E:/LIDC/LIDC-Nodules-Selected/'
    nonNodulePath = 'D:/ProjectData/LIDC/LIDC-NonNodules-CSV/'
    savepath = 'D:/ProjectData/LIDC/Npy-Seperate/OriginCsv/'
    os.makedirs(savepath)

    for appoint in range(10):
        saveData, saveLabel = [], []
        counter = 0
        for indexA in os.listdir(nodulePath):
            print('Loading Nodules :', indexA)
            for indexB in os.listdir(os.path.join(nodulePath, indexA)):
                for indexC in ['Csv']:
                    for indexD in os.listdir(os.path.join(nodulePath, indexA, indexB, indexC)):
                        if counter % 10 == appoint:
                            data = numpy.genfromtxt(fname=os.path.join(nodulePath, indexA, indexB, indexC, indexD),
                                                    dtype=float, delimiter=',')
                            saveData.append(data)
                            saveLabel.append([1, 0])
                        counter += 1
        print(numpy.shape(saveData), numpy.shape(saveLabel))
        # for indexA in os.listdir(nodulePath):
        #     print('Loading Nodules :', indexA)
        #     for indexB in os.listdir(os.path.join(nodulePath, indexA)):
        #         for indexC in os.listdir(os.path.join(nodulePath, indexA, indexB)):
        #             if counter % 10 == appoint:
        #                 data = numpy.genfromtxt(fname=os.path.join(nodulePath, indexA, indexB, indexC), dtype=float,
        #                                         delimiter=',')
        #                 saveData.append(data)
        #                 saveLabel.append([1, 0])
        #             counter += 1

        counter = 0
        for indexA in os.listdir(nonNodulePath):
            print('Loading NonNodules :', indexA)
            for indexB in os.listdir(os.path.join(nonNodulePath, indexA)):
                if counter % 10 == appoint:
                    data = numpy.genfromtxt(fname=os.path.join(nonNodulePath, indexA, indexB), dtype=float,
                                            delimiter=',')
                    saveData.append(data)
                    saveLabel.append([0, 1])
                counter += 1
        print(numpy.shape(saveData), numpy.shape(saveLabel), numpy.sum(saveLabel, axis=0))

        numpy.save(savepath + 'Appoint-%d-Data.npy' % appoint, saveData)
        numpy.save(savepath + 'Appoint-%d-Label.npy' % appoint, saveLabel)
        # exit()
