import numpy as np
from sklearn import cross_validation
from sklearn import datasets
from sklearn import svm

# load test data
iris = datasets.load_iris()
# split data in train set and test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
     iris.data, iris.target, test_size=0.4, random_state=0
)

# use svm 
clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf.score(X_test, y_test)

# cross validation
clf = svm.SVC(kernel='linear', C=1)
# cv : 分割数据集份数，
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
# 可以以此计算此分类器的均值方差，比较分类器之间的表现
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# scoring : 分类器评分标准 String类型
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_weighted')

# 也可以交叉验证得到预测值
predicted = cross_validation.cross_val_predict(clf, iris.data, iris.target, cv=10)
metrics.accuracy_score(iris.target, predicted) 


# 交叉验证迭代器
# KFold类
# n : 元素总数
# n_folds : 交叉验证数量，当n_folds=n时，为留一法 Leave-One-Out
# shuffle : 是否随机打乱
kf = cross_validation.KFold(4, n_folds=4)
for train, test in kf:
    print("%s %s" % (train, test))

# 留一法
# LeaveOneOut类
loo = cross_validation.LeaveOneOut(4)
for train, test in loo:
    print("%s %s" % (train, test))




