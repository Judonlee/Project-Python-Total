from sklearn.svm import SVC
import numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from scipy import interp
from sklearn.metrics import roc_curve, auc
import matplotlib.pylab as plt

if __name__ == '__main__':
    mean_fpr = numpy.linspace(0, 1, 100)
    mean_tpr = 0.0
    part = ['cA', 'cD', 'cH', 'cV']
    # part = ['cV']
    conf = 'db4'
    aucList = []
    for appoint in range(10):
        trainData, testData = [], []
        trainLabel = numpy.load('E:/LIDC/Npy/Wavelet-' + conf + '/Appoint-%d/TrainLabel.npy' % appoint)
        testLabel = numpy.load('E:/LIDC/Npy/Wavelet-' + conf + '/Appoint-%d/TestLabel.npy' % appoint)
        trainLabel = numpy.argmax(trainLabel, axis=1)
        testLabel = numpy.argmax(testLabel, axis=1)

        for sample in part:
            currentData = numpy.load('E:/LIDC/Npy/Wavelet-' + conf + '/Appoint-%d/' % appoint + sample + '-Train.npy')
            currentData = numpy.reshape(currentData,
                                        newshape=[-1, numpy.shape(currentData)[1] * numpy.shape(currentData)[2]])
            if len(trainData) == 0:
                trainData = currentData.copy()
            else:
                trainData = numpy.concatenate((trainData, currentData), axis=1)

            currentData = numpy.load('E:/LIDC/Npy/Wavelet-' + conf + '/Appoint-%d/' % appoint + sample + '-Test.npy')
            currentData = numpy.reshape(currentData,
                                        newshape=[-1, numpy.shape(currentData)[1] * numpy.shape(currentData)[2]])
            if len(testData) == 0:
                testData = currentData.copy()
            else:
                testData = numpy.concatenate((testData, currentData), axis=1)

        print(numpy.shape(trainData), numpy.shape(testData), numpy.shape(trainLabel), numpy.shape(testLabel))

        totalData = numpy.concatenate((trainData, testData), axis=0)
        pca = PCA(n_components=10)
        pca.fit(totalData)
        print(sum(pca.explained_variance_ratio_))

        trainData = pca.transform(trainData)
        testData = pca.transform(testData)

        totalData = numpy.concatenate((trainData, testData), axis=0)
        totalData = scale(totalData)
        trainData = totalData[0:len(trainData)]
        testData = totalData[len(trainData):]
        clf = SVC(probability=True)
        clf.fit(trainData, trainLabel)
        probability = clf.predict_proba(testData)
        fpr, tpr, thresholds = roc_curve(testLabel, probability[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d' % appoint)
        aucList.append(roc_auc)

    name = '' + conf
    for sample in part:
        name = name + '-' + sample

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.legend()
    plt.show()
    for sample in aucList:
        print(sample)
