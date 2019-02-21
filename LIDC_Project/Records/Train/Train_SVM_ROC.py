from LIDC_Project.Records.Loader.LIDC_Loader import LIDC_Loader_Npy
from sklearn.svm import SVC
import numpy
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import scale
from scipy import interp
import matplotlib.pylab as plt

if __name__ == '__main__':
    mean_fpr = numpy.linspace(0, 1, 100)
    mean_tpr = 0.0
    used = 'LBP_P=24_R=3'

    aucList = []
    for appoint in range(10):
        # savepath = 'E:/LIDC/Result-SVM/OriginCsv/Appoint-%d/' % appoint
        # if not os.path.exists(savepath):
        #     os.makedirs(savepath)

        trainData, trainLabel, testData, testLabel = LIDC_Loader_Npy(
            loadpath='E:/LIDC/Npy/' + used + '/Appoint-%d/' % appoint)

        totalNumber = numpy.sum(testLabel, axis=0)[0]

        trainData = numpy.reshape(trainData, newshape=[-1, 4096])
        testData = numpy.reshape(testData, newshape=[-1, 4096])
        trainLabel = numpy.argmax(trainLabel, axis=1)
        testLabel = numpy.argmax(testLabel, axis=1)
        # print(numpy.shape(trainData), numpy.shape(trainLabel), numpy.shape(testData), numpy.shape(testLabel))
        # print(trainData[0:5])
        totalData = numpy.concatenate((trainData, testData), axis=0)
        pca = PCA(n_components=10)
        pca.fit(totalData)

        trainData = pca.transform(trainData)
        testData = pca.transform(testData)

        totalData = numpy.concatenate((trainData, testData), axis=0)
        totalData = scale(totalData)
        trainData = totalData[0:len(trainData)]
        testData = totalData[len(trainData):]
        # print(trainData[0:5])

        clf = SVC(probability=True)
        clf.fit(trainData, trainLabel)
        # joblib.dump(clf, savepath + 'Parameter.m')
        probability = clf.predict_proba(testData)
        fpr, tpr, thresholds = roc_curve(testLabel, probability[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
        mean_tpr[0] = 0.0  # 初始处为0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1, label='ROC fold %d' % appoint)
        aucList.append(roc_auc)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(used)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    plt.legend()
    plt.show()

    for sample in aucList:
        print(sample)
