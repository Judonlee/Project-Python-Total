import numpy
import random


def Shuffle(data, label, seqLen):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel, newSeqLen = [], [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
        newSeqLen.append(seqLen[sample])
    return newData, newLabel, newSeqLen


def Shuffle_Train(data, label):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel = [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
    return newData, newLabel
