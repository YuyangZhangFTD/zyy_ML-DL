import numpy as np
from sklearn import grid_search
from sklearn import svm
from sklearn import datasets
import scipy

iris = datasets.load_iris()
svr = svm.SVC()

# 遍历搜索最佳参数
# GridSearchCV类
parameters = {
    'kernel':('linear', 'rbf'), 
    'C':[1, 10]
}
# 传入参数为分类器和分类器参数字典
clf = grid_search.GridSearchCV(svr, parameters)
clf.fit(iris.data, iris.target)

# 随机搜索最佳参数
# RandomizedSearchCV类
parameters = [
    {
        'C': scipy.stats.expon(scale=100), 
        'gamma': scipy.stats.expon(scale=.1),
        'kernel': ['rbf'], 
        'class_weight':['auto', None]
        }
  ]
clf = grid_search.RandomizedSearchCV(svr, parameters)