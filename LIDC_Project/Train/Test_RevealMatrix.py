import numpy
import os

if __name__ == '__main__':
    loadpath = 'D:/GitHub/LIDC_Project/Train/'
    for part in ['Tree', 'SVC', 'AdaBoost', 'Gaussian']:
        AUCData = numpy.genfromtxt(fname=loadpath + 'Result-AUC-%s.csv' % part, dtype=float, delimiter=',')[:, :-1]
        PrecisionData = numpy.genfromtxt(
            fname=loadpath + 'Result-Precision-%s.csv' % part, dtype=float, delimiter=',')[:, :-1]
        SensitivityData = numpy.genfromtxt(
            fname=loadpath + 'Result-Sensitivity-%s.csv' % part, dtype=float, delimiter=',')[:, :-1]
        SpecificityData = numpy.genfromtxt(
            fname=loadpath + 'Result-Specificity-%s.csv' % part, dtype=float, delimiter=',')[:, :-1]

        print(part)
        # print(numpy.max(AUCData, axis=0))
        for index in range(5):
            print(numpy.max(AUCData, axis=0)[index], '\t', numpy.max(PrecisionData, axis=0)[index], '\t',
                  numpy.max(SensitivityData, axis=0)[index], '\t', numpy.max(SpecificityData, axis=0)[index])
        print('\n\n')

        # exit()
