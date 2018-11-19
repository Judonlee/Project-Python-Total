import os
import numpy

if __name__ == '__main__':
    instancePath = 'E:/LIDC/TreatmentTrace/Step1-InstanceNumber/'
    mediaPosition = 'E:/LIDC/TreatmentTrace/Step2-MediaPosition'
    savepath = 'E:/LIDC/TreatmentTrace/Step3-NoduleMedia/'
    os.makedirs(savepath)

    for indexA in os.listdir(mediaPosition):
        file = open(savepath + indexA + '.csv', 'w')
        instanceDictionary = numpy.genfromtxt(os.path.join(instancePath, indexA + '.csv'), dtype=str, delimiter=',')
        dictionary = {}
        for sample in instanceDictionary:
            dictionary[sample[1]] = int(sample[0])
        # print(dictionary)

        for indexB in os.listdir(os.path.join(mediaPosition, indexA)):
            position = numpy.genfromtxt(os.path.join(mediaPosition, indexA, indexB, 'Position.csv'), dtype=str,
                                        delimiter=',')
            currentLocation = []
            position = numpy.reshape(position, [-1, 3])
            for sample in position:
                if sample[0] not in dictionary.keys():
                    continue
                currentLocation.append([dictionary[sample[0]], int(sample[1]), int(sample[2])])
            # print(numpy.median(currentLocation, axis=0))
            if len(currentLocation) == 0: continue
            file.write(indexB)
            for sample in numpy.median(currentLocation, axis=0):
                file.write(',' + str(sample))
            file.write('\n')
        file.close()
        # exit()
