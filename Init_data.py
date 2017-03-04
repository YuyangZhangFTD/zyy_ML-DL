# author zyyFTD
# Github: https://github.com/YuyangZhangFTD/zyy_ML-DL

"""
    this code is for python3
"""
import numpy as np
import matplotlib.pyplot as plt


# parameter
n_sample = 100
k = 2


cls_n = int(n_sample/k)
# 4 centers
c = [np.array([2, 2]),
     np.array([8, 8]),
     np.array([8, 2]),
     np.array([2, 8])
     ]
sigma = np.array([[1, 0], [0, 1]])
data = np.zeros([n_sample, 2])
cls = np.ones([n_sample, 1])
for ii in range(k):
    data[ii*cls_n:(ii+1)*cls_n] = np.dot(np.random.randn(cls_n, 2), sigma) + c[ii]
    cls[ii*cls_n:(ii+1)*cls_n] *= ii

# plot data
color = ['b', 'y', 'g', 'r']
for ii in range(k):
    plt.scatter(data[ii*cls_n:(ii+1)*cls_n, 0], data[ii*cls_n:(ii+1)*cls_n, 1], c=color[ii])
plt.show()

# save data
with open('Labeled_Data_2cls_100.csv', 'w') as f:
    f.write('x,y,label\n')
    for ii in range(n_sample):
        tmp = ','.join([str(x) for x in data[ii, :].tolist()]) + ','+str(int(cls[ii]))
        f.write(tmp + '\n')
