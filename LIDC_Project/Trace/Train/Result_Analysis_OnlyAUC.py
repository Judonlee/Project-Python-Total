import os
from LIDC_Project.Trace.Train.Tools import AUC_Calculation
import numpy

if __name__ == '__main__':
    searchPath = 'E:/LIDC/TreatmentTrace/Step8-Result/Step8-Result/'
    with open('Result.csv', 'w') as file:
        for indexA in range(1, 200):
            for indexB in range(10):
                print(indexA, indexB)
                testLabel = numpy.load('E:/LIDC/TreatmentTrace/Step7-TotalNpy/OriginCsv/Part%d-Label.npy' % indexB)
                testLabel = numpy.argmax(testLabel, axis=1)

                probability = numpy.genfromtxt(
                    os.path.join('E:/LIDC/TreatmentTrace/Step8-Result/Step8-Result/OriginCsv-SVM-%04d/Batch%d.csv' % (
                        indexA, indexB)), dtype=float, delimiter=',')
                auc = AUC_Calculation(testLabel=testLabel, probability=probability)
                print(auc)
                file.write(str(auc) + ',')
            file.write('\n')
