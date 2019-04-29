import numpy
from sklearn.preprocessing import scale

if __name__ == '__main__':
    loadpath = 'D:/GitHub/DepressionRecognition/Test/None-0-frame-%s.csv'
    savepath = 'E:/ProjectData_Depression/Step5_Assembly/SpeechLevel/None-0-frame-%s.csv'
    trainData = numpy.genfromtxt(loadpath % 'Train', dtype=float, delimiter=',')
    testData = numpy.genfromtxt(loadpath % 'Test', dtype=float, delimiter=',')

    totalData = numpy.concatenate([trainData, testData], axis=0)
    print(numpy.shape(totalData))
    totalData = scale(totalData)

    trainData = totalData[0:len(trainData)]
    testData = totalData[len(trainData):]

    with open(savepath % 'Train', 'w') as file:
        for indexX in range(numpy.shape(trainData)[0]):
            for indexY in range(numpy.shape(trainData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(trainData[indexX][indexY]))
            file.write('\n')

    with open(savepath % 'Test', 'w') as file:
        for indexX in range(numpy.shape(testData)[0]):
            for indexY in range(numpy.shape(testData)[1]):
                if indexY != 0: file.write(',')
                file.write(str(testData[indexX][indexY]))
            file.write('\n')
