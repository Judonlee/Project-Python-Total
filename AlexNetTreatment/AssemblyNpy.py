import numpy

if __name__ == '__main__':
    usedPart = 'Ohm'
    data, label = [], []
    savepath = 'D:/Matlab/VGG19/'
    for index in range(5):
        currentData = numpy.genfromtxt('D:\Matlab\VGG19\%s-%d.csv' % (usedPart, index), dtype=float, delimiter=',')
        currentLabel = numpy.tile([index], [numpy.shape(currentData)[0]])
        print(numpy.shape(currentData), numpy.shape(currentLabel))
        data.extend(currentData)
        label.extend(currentLabel)
    print(numpy.shape(data), numpy.shape(label))
    numpy.save(savepath + '%s-Data.npy' % usedPart, data)
    numpy.save(savepath + '%s-Label.npy' % usedPart, label)
