from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
from scipy import interp
import matplotlib.pylab as plt
import numpy


def ROCPrinter(trainData, trainLabel, testData, testLabel, appoint, C=1, gamma='auto'):
    mean_fpr = numpy.linspace(0, 1, 100)
    mean_tpr = 0.0
    clf = SVC(probability=True, C=C, gamma=gamma)
    clf.fit(trainData, trainLabel)
    probability = clf.predict_proba(testData)
    fpr, tpr, thresholds = roc_curve(testLabel, probability[:, 1])
    mean_tpr += interp(mean_fpr, fpr, tpr)  # 对mean_tpr在mean_fpr处进行插值，通过scipy包调用interp()函数
    mean_tpr[0] = 0.0  # 初始处为0
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='ROC fold %d' % appoint)

    return roc_auc
