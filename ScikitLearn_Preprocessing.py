from sklearn import preprocessing
import numpy as np
import scipy.sparse as sp

X = np.array(
    [
        [1,2,3,4],
        [6,7,8,9],
        [3,4,5,6],
        [7,8,9,1]
    ]
)
# axis 是增长方向
# axis=0 按行增加方向
# axis=1 按列增加方向
# 标准化后均值为0，方差为1
X_scaled = preprocessing.scale(X,axis=1)
 
# StandardScaler类
# copy=True 
# with_mean=True
# with_std=True
scaler = preprocessing.StandardScaler().fit(X)
# 根据X来确定缩放标准，可以用于让训练集与测试集同分布
scaler.transform(X) 

# MinMaxScaler类
# 根据特征的范围缩放
# feature_range=(min, max), 默认[0,1]
min_max_scaler = preprocessing.MinMaxScaler()
X_minmax = min_max_scaler.fit_transform(X)
X_minmax = min_max_scaler.transform(X)

# MaxAbsScaler类
# 将数据缩放到[-1,+1]
# 均值为0
max_abs_scaler = preprocessing.MaxAbsScaler()
X_maxabs = max_abs_scaler.fit_transform(X)

# 矩阵规范化
# norm 规范化的范数
X_normalized = preprocessing.normalize(X, norm='l2')

# 二值化
# Binarizer类
# copy=True
# threshold=0.0
binarizer = preprocessing.Binarizer().fit(X)
binarizer.transform(X)

# OneHot Encoding
# OneHotEncoder类
# n_values 设置每个特征分成几个 输入int类型或者[int,int,..,int]
# n_values='auto' 默认按出现分
enc = preprocessing.OneHotEncoder()
enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
enc.transform([[0, 1, 3]]).toarray()

# 缺失值处理
# Imputer类
# miss_values 需要处理的缺失值
# strategy 处理缺失值的方法 mean median most_frequent
# axis 处理缺失值时的根据行或者列或者别的
imp = preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)
# 需要fit
imp.fit([[1, 2], [np.nan, 3], [7, 6]])
X = [[np.nan, 2], [6, np.nan], [7, 6]]
print(imp.transform(X))
# 处理矩阵
X = sp.csc_matrix([[1, 2], [0, 3], [7, 6]])
imp.fit(X)
# 替换0值
Imputer(axis=0, copy=True, missing_values=0, strategy='mean', verbose=0)
X = sp.csc_matrix([[0, 2], [6, 0], [7, 6]])
print(imp.transform(X))

# 多项式类特征生成
# PolynomialFeatures类
X = np.arange(6).reshape(3, 2)
# degree : integer  最高项阶数
# interaction_only : boolean, default = False 是否只含交叉项
# include_bias : boolean
# 特征向量X从:math:(X_1, X_2) 被转换成:math:(1, X_1, X_2, X_1^2, X_1X_2, X_2^2)
poly = preprocessing.PolynomialFeatures(2)
# fit+transform
poly.fit_transform(X)

# 数据预处理转换器
# FunctionTransformer类
# 传参为自定义的函数
transformer = preprocessing.FunctionTransformer(lambda x: x+1)
X = np.array([[0, 1], [2, 3]])
print(transformer.transform(X))

