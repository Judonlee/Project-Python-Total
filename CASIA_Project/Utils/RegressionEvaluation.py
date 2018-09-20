import numpy


def MAE_Calculation(predict, label):
    total = 0
    if len(predict) != len(label):
        print('ERROR : predict and label don"t have same length.')
    for index in range(len(predict)):
        total += numpy.abs(predict[index] - label[index])
    total = total / len(predict)
    return total


def RMSE_Calculation(predict, label):
    total = 0
    if len(predict) != len(label):
        print('ERROR : predict and label don"t have same length.')
    for index in range(len(predict)):
        total += (predict[index] - label[index]) * (predict[index] - label[index])
    total = total / len(predict)
    return numpy.sqrt(total)
