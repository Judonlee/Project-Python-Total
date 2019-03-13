import numpy
import os

if __name__ == '__main__':
    for part in ['AdaBoost', 'Gaussian', 'SVC', 'Tree']:
        loadpath = 'E:/ProjectData_LIDC/Features/Step4_Result/R=3_P=24_DX/%s/' % part
        filePrecision = open('Result-Precision-%s.csv' % part, 'w')
        fileSensitivity = open('Result-Sensitivity-%s.csv' % part, 'w')
        fileSpecificity = open('Result-Specificity-%s.csv' % part, 'w')

        for filename in os.listdir(loadpath):
            print(filename, filename[18])
            data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',')

            labelData = numpy.genfromtxt(
                fname='E:/ProjectData_LIDC/Features/Step4_Result/Featurelabel_%s.csv' % filename[18], dtype=int,
                delimiter=',')

            matrix = numpy.zeros([2, 2])
            for index in range(len(data)):
                matrix[1 - labelData[index]][numpy.argmax(data[index])] += 1

            filePrecision.write(str((matrix[0][0] + matrix[1][1]) / sum(sum(matrix))) + ',')
            fileSensitivity.write(str(matrix[0][0] / sum(matrix[0])) + ',')
            fileSpecificity.write(str(matrix[1][1] / sum(matrix[1])) + ',')

            if filename[18] == '4':
                filePrecision.write('\n')
                fileSensitivity.write('\n')
                fileSpecificity.write('\n')

            # print(matrix)
            # exit()
