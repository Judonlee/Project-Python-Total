import numpy
import os
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc

if __name__ == '__main__':
    loadpath = 'E:/ProjectData_LIDC/Features/Step4_Result/SVC/'
    with open('Result-AUC.csv', 'w') as file:
        for filename in os.listdir(loadpath):
            # print(filename, filename[18])
            data = numpy.genfromtxt(fname=os.path.join(loadpath, filename), dtype=float, delimiter=',')

            labelData = numpy.genfromtxt(
                fname='E:/ProjectData_LIDC/Features/Step4_Result/Featurelabel_%s.csv' % filename[18], dtype=int,
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
                print(filename)
                os.remove(os.path.join(loadpath, filename))

        # plt.plot(fpr, tpr, marker='o')
        # plt.show()

        # exit()
