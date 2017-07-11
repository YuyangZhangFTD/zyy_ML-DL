# author zyyFTD
# Github: https://github.com/YuyangZhangFTD/zyy_ML-DL

"""
    this code is for python3

    neural network with tensorflow
"""


import numpy as np
import tensorflow as tf
import pandas as pd


# parameter init
n_epochs = 100
eta = 0.01
n_input = 2
n_hidden1 = 5
n_hidden2 = 5
n_output = 4


# ========================================== function ====================================================
# You can write your own neural network model here.
# It is not a good style to define a model in this way.
# Define keys of weights and biases in a convenient way, then use loop to define each layer.
# By the way, batch normalization and other tricks can be applied here.
def my_nn(para_x, para_weights, para_biases):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(para_x, para_weights['h1']), para_biases['b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, para_weights['h2']), para_biases['b2']))
    out_layer = tf.nn.softmax(tf.matmul(layer_2, para_weights['out']) + para_biases['out'])
    return out_layer


def one_hot(para_y, para_output):        # label one-hot encoding [2]==>[0,1,0,0,0,0,0,0,0,0]
    tmp = []
    for ii in range(len(para_y)):
        data = [0] * para_output
        data[int(para_y[ii])] = 1
        tmp.append(data)
    return np.mat(tmp)


def one_encode(para_x):     # the input data [1-256]==>[1]
    tmp = []
    for ii in range(len(para_x)):
        tmp.append(list(map(lambda x: 1 if x > 0 else 0, para_x[ii])))
    return np.mat(tmp)
# =======================================================================================================

'''
# classification test
# ========================================== init parameter ==============================================
# read data
file = pd.read_csv('Labeled_Data_4cls_100.csv')
ys = one_hot(file['label'].values, n_output)
xs = np.mat(file.drop('label', axis=1))


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_output])),
}

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

y_hat = my_nn(X, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=Y))
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for i_epoch in range(n_epochs):
        for (x, y) in zip(xs, ys):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        if i_epoch % 20 == 0:
            training_cost = sess.run(accuracy, feed_dict={X: xs, Y: ys})
            print(training_cost)
'''


# mnist test
# ========================================== init parameter ==============================================
learning_rate = 0.01
n_input = 784
n_hidden1 = 256
n_hidden2 = 64
n_output = 10
epoch_n = 100
batch_size = 512             # batch size should be 2^n, for GPU training.
# ========================================================================================================

# read data
# train data
train = pd.read_csv('mnist_train_mine.csv')
train_y = one_hot(train.label.values, n_output)
train_x = one_encode(train.drop('label', 1).values)
n_train = len(train_x)
# test data
test = pd.read_csv('mnist_test_mine.csv')
test_y = one_hot(test.label.values, n_output)
test_x = one_encode(test.drop('label', 1).values)


weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_output])),
}

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

y_hat = my_nn(X, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=Y))
correct_prediction = tf.equal(tf.argmax(y_hat, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
optimizer = tf.train.AdamOptimizer(learning_rate=eta).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_training_cost = 0.0
    for i_epoch in range(n_epochs):
        total_batch = int(n_train / batch_size)
        for i in range(total_batch):
            avg_cost = 0.
            batch_index = np.random.randint(n_train, size=[batch_size])
            batch_x = train_x[batch_index]
            batch_y = train_y[batch_index]
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
        if i_epoch % 20 == 0:
            training_cost = sess.run(accuracy, feed_dict={X: train_x, Y: train_y})
            print(training_cost)

    print("Accuracy:", accuracy.eval({X: test_x, Y: test_y}))