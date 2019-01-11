import numpy
import os

if __name__ == '__main__':
    loadpathA = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands40/'
    loadpathB = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands40-CNN/'
    for filename in os.listdir(loadpathA):
        if filename[-8:] != 'Data.npy': continue
        dataA = numpy.load(loadpathA + filename)
        dataB = numpy.load(loadpathB + filename)
        print(filename)
        for indexX in range(len(dataA)):
            for indexY in range(len(dataA[indexX])):
                for indexZ in range(len(dataA[indexX][indexY])):
                    if dataA[indexX][indexY][indexZ] != dataB[indexX][indexY][indexZ]:
                        print('ERROR')
                        exit()
