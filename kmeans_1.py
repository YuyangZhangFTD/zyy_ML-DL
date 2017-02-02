# author zyyFTD
# Github: https://github.com/YuyangZhangFTD/zyy_ML-DL

"""
    this code is for python3
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# parameter
k = 2
n_epoch = 60

# load data
data = pd.read_csv('Unlabeled_Data_2cls_1000.csv').values
x = data[:, 0]
y = data[:, 1]
n_sample = len(data)
cls_n = int(n_sample/k)

# init center points
c = np.random.uniform(0, 10, [k, 2])
cls = np.ones([n_sample, 1])

# plot data
color = ['g', 'r', 'b', 'y', ]

# plot data
plt.ion()
fig, ax = plt.subplots(1, 1)
ax.scatter(x, y)
for i in range(k):
    ax.scatter(x=c[i, 0], y=c[i, 1], s=40, c='r')
fig.show()
plt.draw()
plt.waitforbuttonpress()

for i_epoch in range(n_epoch):
    # E-step
    # calculate dis, assign the point to center point
    dis = np.zeros([n_sample, k])
    for i in range(n_sample):
        for j in range(k):
            dis[i, j] = np.sqrt(np.sum(np.power(c[j, :] - data[i, :], 2)))
        cls[i] = dis[i, :].argmin()

    # M-step
    # re-calculate center point position
    for j in range(k):
        c[j, :] = np.average([dis[x, j] for x in np.argwhere(cls == j).tolist()], axis=0)
        print(j)

    # plot
    if i_epoch % 20 == 0:
        for i in range(n_sample):
            ax.scatter(x[i], y[i], c=color[int(cls[i])])
        for i in range(k):
            ax.scatter(x=c[i, 0], y=c[i, 1], s=40, c='b')
        fig.show()
        plt.draw()
        plt.waitforbuttonpress()
