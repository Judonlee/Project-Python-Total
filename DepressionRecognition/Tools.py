import numpy
import random


def Shuffle_Part3(data, label, seq):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel, newSeq = [], [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
        newSeq.append(seq[sample])
    return newData, newLabel, newSeq


def Shuffle(data, label, dataLen, labelLen):
    index = numpy.arange(0, len(data))
    random.shuffle(index)
    newData, newLabel, newDataLen, newLabelLen = [], [], [], []
    for sample in index:
        newData.append(data[sample])
        newLabel.append(label[sample])
        newDataLen.append(dataLen[sample])
        newLabelLen.append(labelLen[sample])
    return newData, newLabel, newDataLen, newLabelLen


def MAE_Calculation(label, predict):
    counter = 0
    for index in range(len(label)):
        counter += numpy.abs(label[index] - predict[index])
    return counter / len(label)


def RMSE_Calculation(label, predict):
    counter = 0
    for index in range(len(label)):
        counter += (label[index] - predict[index]) * (label[index] - predict[index])
    return numpy.sqrt(counter / len(label))
