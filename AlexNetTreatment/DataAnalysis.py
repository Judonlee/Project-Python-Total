import numpy

data = numpy.load(r'E:\OneUseData\FAU\bands-30\ohm-std.npy')
label = numpy.load(r'E:\OneUseData\FAU\bands-30\ohm-label.npy')
# for sample in data:
#     print(numpy.shape(sample))
print(numpy.shape(label))
print(numpy.sum(label, axis=0))

print(numpy.sum(label) / (5 * numpy.bincount(numpy.argmax(label, axis=1))))
