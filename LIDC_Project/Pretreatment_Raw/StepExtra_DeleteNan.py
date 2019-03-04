import numpy

if __name__ == '__main__':
    datafile = 'E:/ProjectData_LIDC/Features/Step0_Raw/DicFeature_%d.csv'
    labelfile = 'E:/ProjectData_LIDC/Features/Step0_Raw/Featurelabel_%d.csv'

    savedatafile = 'E:/ProjectData_LIDC/Features/Step0_Raw/DicFeature_Restart_%d.csv'
    savelabelfile = 'E:/ProjectData_LIDC/Features/Step0_Raw/Featurelabel_Restart_%d.csv'

    for index in range(5):
        labeldata = numpy.genfromtxt(fname=labelfile % index, dtype=int, delimiter=',')
        with open(datafile % index, 'r') as file:
            data = file.readlines()
        with open(savedatafile % index, 'w') as fileData:
            with open(savelabelfile % index, 'w') as fileLabel:
                for index in range(len(data)):
                    if data[index].find('NaN') != -1: continue

                    fileData.write(data[index])
                    fileLabel.write(str(labeldata[index]) + '\n')
