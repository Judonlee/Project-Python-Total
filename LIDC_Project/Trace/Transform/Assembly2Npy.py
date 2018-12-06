import os
import numpy

if __name__ == '__main__':
    for part in ['cH', 'cV']:
        nodulePath = 'E:/LIDC/TreatmentTrace/Step5-NodulesCsv-Seperate/'
        nonNodulePath = 'E:/LIDC/TreatmentTrace/Step5-NonNodulesCsv/'
        savePath = 'E:/LIDC/TreatmentTrace/Step7-TotalNpy/Wavelet_db2/' + part + '/'
        if not os.path.exists(savePath): os.makedirs(savePath)

        for appoint in range(10):
            partData, partLabel = [], []
            counter = 0
            for indexA in os.listdir(nodulePath):
                print(part, appoint, indexA)
                for indexB in os.listdir(os.path.join(nodulePath, indexA)):
                    for indexC in os.listdir(os.path.join(nodulePath, indexA, indexB)):
                        # print(indexA, indexB, indexC)
                        if counter % 10 == appoint:
                            data = numpy.genfromtxt(fname=os.path.join(nodulePath, indexA, indexB, indexC), dtype=float,
                                                    delimiter=',')
                            label = [1, 0]
                            partData.append(data)
                            partLabel.append(label)
                        counter += 1

            counter = 0
            for indexA in os.listdir(nonNodulePath):
                print(part, indexA)
                for indexB in os.listdir(os.path.join(nonNodulePath, indexA)):
                    if counter % 10 == appoint:
                        data = numpy.genfromtxt(fname=os.path.join(nonNodulePath, indexA, indexB), dtype=float,
                                                delimiter=',')
                        label = [0, 1]
                        partData.append(data)
                        partLabel.append(label)
                    counter += 1

            print(numpy.shape(partData), numpy.shape(partLabel), numpy.sum(partLabel, axis=0))
            numpy.save(os.path.join(savePath, 'Part%d-Data.npy' % appoint), partData)
            numpy.save(os.path.join(savePath, 'Part%d-Label.npy' % appoint), partLabel)
