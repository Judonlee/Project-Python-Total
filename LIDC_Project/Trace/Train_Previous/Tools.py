import numpy
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pylab as plt
import multiprocessing as mp
import time


def LoadPart(loadpath, appoint):
    trainData, trainLabel, testData, testLabel = [], [], [], []
    for counter in range(10):
        data = numpy.load(os.path.join(loadpath, 'Part%d-Data.npy' % counter))
        label = numpy.load(os.path.join(loadpath, 'Part%d-Label.npy' % counter))
        # print(numpy.shape(data), numpy.shape(label))

        if counter == appoint:
            testData.extend(data)
            testLabel.extend(label)
        else:
            trainData.extend(data)
            trainLabel.extend(label)
    print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.sum(trainLabel, axis=0))
    print(numpy.shape(testData), numpy.shape(testLabel), numpy.sum(testLabel, axis=0))
    return trainData, trainLabel, testData, testLabel


def PCA_Treatment(trainData, testData, componentNumber):
    totalData = numpy.concatenate((trainData, testData), axis=0)
    pca = PCA(n_components=componentNumber)
    pca.fit(totalData)
    trainData = pca.transform(trainData)
    testData = pca.transform(testData)

    return Normalization_Treatment(trainData=trainData, testData=testData)


def Normalization_Treatment(trainData, testData):
    totalData = numpy.concatenate((trainData, testData), axis=0)
    totalData = scale(totalData)
    trainData = totalData[0:len(trainData)]
    testData = totalData[len(trainData):]
    return trainData, testData


def AUC_Calculation(testLabel, probability, figureFlag=False, showFlag=False, legend=''):
    mean_fpr = numpy.linspace(0, 1, 100)
    mean_tpr = 0.0

    fpr, tpr, thresholds = roc_curve(testLabel, probability[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr[0] = 0.0  # 初始处为0
    roc_auc = auc(fpr, tpr)

    # print(roc_auc)
    if figureFlag:
        if legend != '':
            plt.plot(fpr, tpr, lw=1, label=legend)
        else:
            plt.plot(fpr, tpr, lw=1)
        if showFlag: plt.show()

    return roc_auc


def Result_Calculation(testLabel, probability):
    matrix = numpy.zeros((numpy.shape(probability)[1], numpy.shape(probability)[1]))
    for index in range(len(testLabel)):
        matrix[testLabel[index]][numpy.argmax(numpy.array(probability[index]))] += 1
    print(matrix)

    Precision = 0
    Sensitivity = matrix[1][1] / sum(matrix[1])
    Specificity = matrix[1][1] / sum(matrix[:, 1])
    for index in range(len(matrix)):
        Precision += matrix[index][index]
    Precision /= sum(sum(matrix))
    return [Precision, Sensitivity, Specificity], matrix


def MP_Treatment(function, times=10):
    processList = []
    for _ in range(times):
        process = mp.Process(target=function)
        process.start()
        processList.append(process)
        time.sleep(5)

    for process in processList:
        process.join()


def DXSingleCalculation(data, label):
    listZero, listOne = [], []
    for index in range(len(data)):
        if label[index] == 0: listZero.append(data[index])
        if label[index] == 1: listOne.append(data[index])
    DXScore = (numpy.mean(listOne) - numpy.mean(listZero)) * (numpy.mean(listOne) - numpy.mean(listZero)) / \
              (numpy.std(listOne) * numpy.std(listOne) + numpy.std(listZero) * numpy.std(listZero))
    return DXScore


def DXFeatureSelection(trainData, trainLabel, testData, testLabel, maxFeatures=10):
    data = numpy.concatenate((trainData, testData), axis=0)
    label = numpy.concatenate((trainLabel, testLabel), axis=0)

    totalScore = []
    for index in range(numpy.shape(data)[1]):
        print('\rTreating %d/%d' % (index, numpy.shape(data)[1]), end='')
        totalScore.append(DXSingleCalculation(data=data[:, index], label=label))

    print('\n')
    featuresSelected = []
    for _ in range(maxFeatures):
        featuresSelected.append(numpy.argmax(totalScore))
        totalScore[numpy.argmax(totalScore)] = 0
    print(featuresSelected)

    results = []
    for indexX in range(numpy.shape(data)[0]):
        current = []
        for indexY in featuresSelected:
            current.append(data[indexX][indexY])
        results.append(current)

    results = scale(results)

    print(numpy.shape(results))
    trainData = results[0:len(trainData)]
    testData = results[len(trainData):]
    return trainData, testData


def DX_Appoint(trainData, testData, featureShape, weights):
    totalData = numpy.concatenate((trainData, testData), axis=0)
    treatedData = []
    for _ in range(featureShape):
        if len(treatedData) == 0:
            treatedData = totalData[:, numpy.argmax(weights):numpy.argmax(weights) + 1].copy()
        else:
            treatedData = numpy.concatenate(
                (treatedData, totalData[:, numpy.argmax(weights):numpy.argmax(weights) + 1]), axis=1)
        weights[numpy.argmax(weights)] = 0
    print(numpy.shape(treatedData))
    treatedData = scale(treatedData)
    trainData = treatedData[0:len(trainData)]
    testData = treatedData[len(trainData):]
    return trainData, testData
