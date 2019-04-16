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


def Shuffle_Four(a, b, c, d):
    index = numpy.arange(0, len(a))
    random.shuffle(index)

    newA, newB, newC, newD = [], [], [], []
    for sample in index:
        newA.append(a[sample])
        newB.append(b[sample])
        newC.append(c[sample])
        newD.append(d[sample])
    return newA, newB, newC, newD


def Shuffle_Five(a, b, c, d, e):
    index = numpy.arange(0, len(a))
    random.shuffle(index)

    newA, newB, newC, newD, newE = [], [], [], [], []
    for sample in index:
        newA.append(a[sample])
        newB.append(b[sample])
        newC.append(c[sample])
        newD.append(d[sample])
        newE.append(e[sample])
    return newA, newB, newC, newD, newE
