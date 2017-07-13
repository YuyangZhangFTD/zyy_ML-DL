import numpy as np
from sklearn import datasets


# load data
# Iris
iris = datasets.load_iris()
# x:(150,4) y:(150,)
x = iris.data
y = iris.target
# Digits
digits = datasets.load_digits()
# x:(1797,8,8) y:(1797,)
# x = digits.images
# x:(1797,64) y:(1797,)
x = digits.images.reshape((digits.images.shape[0], -1))
y = digits.target