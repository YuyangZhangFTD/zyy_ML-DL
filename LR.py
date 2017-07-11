# author zyyFTD
# Github: https://github.com/YuyangZhangFTD/zyy_ML-DL

"""
    This code is for python3
    Logistic Regression
"""

import numpy as np
import pandas as pd

# parameter init
learning_rate = 0.001
n_epochs = 100


def sigmod(para_x, para_w):
	wx = np.array([np.dot(x,para_w) for x in para_x])
	return 1/(1+np.exp((-1)*wx))


def ave_loss(para_x, para_w, para_y):
	wx = np.array([np.dot(x,para_w) for x in para_x])
	tmp = np.exp(wx+1)
	return (-1)*np.average(para_y*wx-np.log(tmp))


def gradient(para_x, para_w, para_y):
	wx = np.array([np.dot(x,para_w) for x in para_x])
	tmp = para_y - 1/(1+np.exp((-1)*wx))
	return (-1)*sum([para_x[ii]*tmp[ii] for ii in range(len(tmp))])


'''
	data (x_1, x_2, 1)	
	label y				\in {0,1}
	weight (w_1, w_2, b)
'''
data = pd.read_csv('Labeled_Data_2CLS_100.CSV')
cls = data.values[:,2]
data = data.values[:,:2]
x_bias = np.ones(len(data))
data = np.column_stack((data, x_bias))
weight = np.random.random(size=3)

for i in range(n_epochs):
		print('epoch :', i)
		print('average loss :', ave_loss(data, weight, cls))
		weight -= gradient(data, weight, cls) * learning_rate
		


