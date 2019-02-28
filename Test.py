import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc

y = np.array([1, 1, 2, 3])
# y为数据的真实标签

scores = np.array([0.1, 0.2, 0.35, 0.8])

# scores为分类其预测的得分

fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)
plt.plot(fpr, tpr)

plt.show()
