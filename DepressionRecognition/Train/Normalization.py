import numpy
import os
from DepressionRecognition.Loader import Load_DBLSTM
from sklearn.preprocessing import scale

if __name__ == '__main__':
    usedpart = 'LA-1-frame'
    loadpath = 'D:/GitHub/DepressionRecognition/Train/%s-%s-First/'
    totalData = []
    for index in range(142):
        data = numpy.load(loadpath % (usedpart, 'Train') + '%04d.npy' % index)
        totalData.extend(numpy.reshape(data, [-1, 256]))
        print('Train', index, numpy.shape(totalData))
    for index in range(47):
        data = numpy.load(loadpath % (usedpart, 'Test',) + '%04d.npy' % index)
        totalData.extend(numpy.reshape(data, [-1, 256]))
        print('Test', index, numpy.shape(totalData))

    totalData = scale(totalData)
    os.makedirs(loadpath % (usedpart + '-Normalization', 'Train'))
    os.makedirs(loadpath % (usedpart + '-Normalization', 'Test'))

    startPosition = 0
    for index in range(142):
        data = numpy.load(loadpath % (usedpart, 'Train') + '%04d.npy' % index)
        numpy.save(loadpath % (usedpart + '-Normalization', 'Train') + '%04d.npy' % index,
                   numpy.reshape(totalData[startPosition:startPosition + numpy.shape(data)[0] * numpy.shape(data)[1]],
                                 [numpy.shape(data)[0], numpy.shape(data)[1], 256]))
        startPosition += numpy.shape(data)[0] * numpy.shape(data)[1]
        print('Save Train', index)
    for index in range(47):
        data = numpy.load(loadpath % (usedpart, 'Test') + '%04d.npy' % index)
        numpy.save(loadpath % (usedpart + '-Normalization', 'Test') + '%04d.npy' % index,
                   numpy.reshape(totalData[startPosition:startPosition + numpy.shape(data)[0] * numpy.shape(data)[1]],
                                 [numpy.shape(data)[0], numpy.shape(data)[1], 256]))
        startPosition += numpy.shape(data)[0] * numpy.shape(data)[1]
        print('Save Test', index)
