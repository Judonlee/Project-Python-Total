import numpy
import os

if __name__ == '__main__':
    loadpathA = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands40/'
    loadpathB = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands40-CNN/'
    savepath = 'E:/ProjectData_SpeechRecognition/Features/IEMOCAP-Npy/Bands40-Seq/'
    os.makedirs(savepath)
    for indexA in ['Female', 'Male']:
        for indexB in range(1, 6):
            print(indexA, indexB)
            dataA = numpy.load(loadpathA + '%s-Session%d-Label.npy' % (indexA, indexB))
            dataB = numpy.load(loadpathB + '%s-Session%d-Label.npy' % (indexA, indexB))

            totalData = []
            for index in range(len(dataA)):
                currentData = numpy.ones(len(dataA[index])) * numpy.argmax(numpy.array(dataB[index]))
                totalData.append(currentData)
            print(numpy.shape(totalData))
            numpy.save(savepath + '%s-Session%d-Label.npy' % (indexA, indexB), totalData)
            # print(currentData)
