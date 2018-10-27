from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import AdaBoostClassifier

iris = load_iris()  # 还是那个数据集
clf = AdaBoostClassifier(n_estimators=100)  # 迭代100次
scores = cross_val_score(clf, iris.data, iris.target)  # 分类器的精确度
print(scores)
