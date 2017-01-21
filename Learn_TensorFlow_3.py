# author zyyFTD
# Github: https://github.com/YuyangZhangFTD/zyy_ML-DL

"""
    this code is for python3
"""


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# initial data
plt.ion()
n_observations = 100
fig, ax = plt.subplots(1, 1)
xs = np.linspace(-3, 3, n_observations)
w = 5
b = 1
ys = w*xs + b + np.random.normal(0, 1, n_observations)
ax.scatter(xs, ys)
fig.show()
plt.draw()
plt.waitforbuttonpress()


# ============================== the normal method ================================
'''
n_epochs = 1000
eta = 0.01


def error(y, y_hat):
    return 1/2*sum((y-y_hat)**2) / (n_observations - 1)


def gradient_w(x, y, y_hat):
    return sum((y-y_hat) * (-x)) / (n_observations - 1)


def gradient_b(x, y, y_hat):
    return sum((y-y_hat) * (-1)) / (n_observations - 1)

w_hat1 = np.random.uniform(0, 10, 1)
b_hat1 = np.random.uniform(0, 10, 1)

for i_epoch in range(n_epochs):
    y_hat1 = w_hat1 * xs + b_hat1
    w_hat1 -= eta * gradient_w(xs, ys, y_hat1)
    b_hat1 -= eta * gradient_b(xs, ys, y_hat1)

    if i_epoch % 20 == 0:
        print(i_epoch, error(ys, y_hat1))
        ax.plot(xs, y_hat1,
                'k', alpha=i_epoch / n_epochs)
        fig.show()
        plt.draw()

print(w_hat1, b_hat1)
fig.show()
plt.waitforbuttonpress()
'''
# =================================== end ====================================

# ============================== with TensorFlow =============================
n_epochs = 1000
eta = 0.01

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w_hat2 = tf.Variable(tf.random_normal([1]), name='weight')
b_hat2 = tf.Variable(tf.random_normal([1]), name='bias')
y_hat2 = tf.add(tf.mul(X, w_hat2), b_hat2)
cost = tf.reduce_sum(tf.pow(y_hat2 - Y, 2)) / (n_observations - 1)
optimizer = tf.train.GradientDescentOptimizer(eta).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for i_epoch in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if i_epoch % 20 == 0:
            training_cost = sess.run(cost, feed_dict={X: xs, Y: ys})
            print(training_cost)
            # ax.plot(xs, y_hat2.eval(feed_dict={x: xs}, session=sess), 'r', alpha=i_epoch / n_epochs)
            fig.show()
            plt.draw()

    print(sess.run(w_hat2), sess.run(b_hat2))
fig.show()
plt.waitforbuttonpress()





