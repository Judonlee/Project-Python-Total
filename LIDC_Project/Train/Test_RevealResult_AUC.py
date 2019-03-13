import numpy
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc

if __name__ == '__main__':
    for part in ['AdaBoost', 'Gaussian', 'SVC', 'Tree']:
        print('Treating %s' % part)
        loadpath = 'E:/ProjectData_LIDC/Features/Step4_Result/R=3_P=24_DX/%s/' % part
        with open('Result-AUC-%s.csv' %  part, 'w') as file:
            for filename in os.listdir(loadpath):
                # print(filename, filename[18])
                try:
                    data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',')
                except:
                    os.remove(os.path.join(loadpath, filename))
                    print('Delete', filename)
                    continue

                labelData = numpy.genfromtxt(
                    fname='E:/ProjectData_LIDC/Features/Step4_Result/Featurelabel_%s.csv' % filename[18],
                    dtype=int,
                    delimiter=',')
                score = data[:, 0]
                try:
                    fpr, tpr, thresholds = metrics.roc_curve(labelData, score, pos_label=1)
                    AUC = auc(fpr, tpr)
                    file.write(str(AUC) + ',')
                    # print(AUC, end='\t')
                    if filename[18] == '4':
                        # print()
                        file.write('\n')
                except:
                    print('AUC Delete', filename)
                    os.remove(os.path.join(loadpath, filename))

            # plt.plot(fpr, tpr, marker='o')
            # plt.show()

            # exit()
