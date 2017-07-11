# author zyyFTD
# Github: https://github.com/YuyangZhangFTD/zyy_ML-DL

"""
    This code is for python3.
    Neural Network using numpy.
    For convenience, all matrices and vectors should be np.mat.

    ================================== The whole architecture ===========================================
    input layer     :           x1      x2      x3      ...     x784    ==>     x[784, 1]
        ||                          full connection                     ==>     w1[512, 784] + b1[512, 1]
    hidden layer1   :           a1      a2      a3      ...     a512    ==>     a[512, 1]
        ||                          full connection                     ==>     w2[256, 512] + b2[256, 1]
    hidden layer2   :           b1      b2      b3      ...     b256    ==>     b[256, 1]
        ||                          full connection                     ==>     w3[128, 256] + b3[128, 1]
    hidden layer3   :           c1      c2      c3      ...     c128    ==>     c[128, 1]
        ||                          full connection                     ==>     w4[64, 128] + b4[64, 1]
    hidden layer4   :           d1      d2      d3      ...     d64     ==>     d[64, 1]
        ||                            softmax                           ==>     w5[64, 10] + b5[10, 1]
    output layer    :           y1      y2      y3      ...     y10     ==>     y[10, 1]

    ====================================== One part of the network ======================================
    output   f()    weight      input  bias
    |y1|        |w11 w12 w13|   |x1|   |b1|
    |y2|  =  f( |w21 w22 w23| * |x2| + |b2| )       f() is activation function
    |y3|        |w31 w32 w33|   |x3|   |b3|

    The correction is C. If the loss function is \frac{1}{2}(\hat{y}-y)^2, C = \hat{y}-y.
    f' is the derivation of f.
    the gradient:
        dC/dw = C*f'*x
        dC/db = C*f'*1
        dC/dx = C*f'*w
    The weight and bias can be updated with gradient of w and b.
    And the correction of the previous layer is the gradient of x in this layer.

    ========================================== softmax ==================================================
    output f()    weight      input   bias
    |y1|      |w11 w12 w13|   |x1|   |b1|
    |y2| = f( |w21 w22 w23| * |x2| + |b2| )       f() is activation function
    |y3|      |w31 w32 w33|   |x3|   |b3|
    |z1| = y1/sum(y)      |y1|
    |z2| = y2/sum(y)    y=|y2|
    |z3| = y3/sum(y)      |y3|

"""
import numpy as np
import pandas as pd


# ================================= init parameter ========================================
layer_n = 3
input_size = 2
hidden1_size = 5
hidden2_size = 5
# hidden3_size = 128
hidden4_size = 5
output_size = 4
learning_rate = 0.1
# batch_size = 32             # batch size should be 2^n, for GPU training.
epoch_n = 1001
# =========================================================================================


# ================================= all function ==========================================
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


def sigmoid(para_x, para_weight, para_bias):
    tmp = np.exp(-1*(para_weight * para_x + para_bias))
    return 1/(1+tmp)


def relu(para_x, para_weight, para_bias):
    tmp = para_weight * para_x + para_bias
    return np.mat(list(map(lambda x: x if x > 0 else 0, [float(x) for x in tmp]))).T


def softmax(para_x, para_weight, para_bias):
    tmp = np.exp(para_weight * para_x + para_bias)
    return tmp/sum(tmp)


def feedforward(para_weight, para_bias, para_value, para_input_vector,
                activation_function=sigmoid, output_function=softmax):
    # the keys of dict must be ordered
    n = len(para_weight.keys())    # get number of layers
    for ii in range(n):
        # get x, w, b for each layer
        if ii == 0:
            tmp_x = para_input_vector
        else:
            tmp_x = para_value[ii-1]
        tmp_weight = para_weight[ii]
        tmp_bias = para_bias[ii]
        # y = f(wx+b)
        # the output layer is softmax layer
        if ii < n-1:
            para_value[ii] = activation_function(tmp_x, tmp_weight, tmp_bias)
        else:
            para_value[ii] = output_function(tmp_x, tmp_weight, tmp_bias)
    return para_value[n-1]


def loss_func(para_hat, para_true):
    # a row vector is a piece of data
    # para_hat    :   [0.1, 0.2, ..., 0.1]
    # para_true   :   [ 0,   1,  ...,  0 ]
    # the sum of a row vector is 1.0
    # calculate the loss of the whole data, return loss and average loss
    return np.sum(np.power((para_hat - para_true), 2) * 0.5)


def backpropagation(para_weight, para_bias, para_value, para_true, para_input_vector,
                    para_eta=0.01, activation_function=sigmoid, loss_function=loss_func):
    n = len(para_weight.keys())  # get number of layers
    # the output layer
    # the derivation of loss function
    if loss_function != loss_func:
        # get the derivation of loss function which you set.
        # error =
        print('Define your own derivation of loss function')
        return None
    else:
        error = para_value[n-1]-para_true       # column vector
    para_weight[n-1] -= para_eta * error * para_value[n-2].T
    para_bias[n-1] -= para_eta * error
    tmp_delta = (para_eta * error.T * para_weight[n-1]).T

    # the hidden layer
    # the derivation of activation function
    if activation_function == sigmoid:
        def gradient(para_para_x):
            return np.multiply(para_para_x, (1 - para_para_x))
    elif activation_function == relu:
        gradient = one_encode
    else:
        # get the derivation of activation function which you set
        # gradient =
        print('Define your own derivation of activation function')
        return None

    for ii in range(n-1)[::-1]:
        tmp_delta = np.multiply(gradient(para_value[ii]), tmp_delta)
        if ii == 0:
            # para_weight[ii] -= para_eta * para_input_vector * tmp_delta.T
            para_weight[ii] -= para_eta * tmp_delta * para_input_vector.T
        else:
            # para_weight[ii] -= para_eta * para_value[ii-1] * tmp_delta.T
            para_weight[ii] -= para_eta * tmp_delta * para_value[ii-1].T
        para_bias[ii] -= para_eta * tmp_delta
        tmp_delta = para_eta * (tmp_delta.T * para_weight[ii]).T
    return para_weight, para_bias


def is_true(para_hat, para_true):
    if np.argmax(para_hat) == np.argmax(para_true):
        return 1
    else:
        return 0


def accuracy(para_hat, para_true):
    true_num = 0
    tmp = 0
    for ii in range(len(para_hat)):
        if np.argmax(para_hat[ii]) == np.argmax(para_true[ii]):
            true_num += 1
        tmp += is_true(para_hat[ii], para_true[ii])
    print(tmp)
    return true_num / len(para_hat)
# ========================================================================================


file = pd.read_csv('Labeled_Data_4cls_100.csv')
train_y = one_hot(file['label'].values, output_size)
train_x = np.mat(file.drop('label', axis=1))

# size_list = [input_size, hidden1_size, hidden2_size, hidden4_size, output_size]
size_list = [input_size, hidden2_size, hidden4_size, output_size]
# size_list = [input_size, hidden4_size, output_size]
weight = dict()
bias = dict()
tmp_value = dict()
for i in range(layer_n):
    weight[i] = np.mat(np.random.rand(size_list[i+1], size_list[i]))
    bias[i] = np.mat(np.random.rand(size_list[i+1], 1))
    tmp_value[i] = np.mat(np.random.rand(size_list[i+1], 1))

data_n = len(train_y)
for epoch_i in range(epoch_n):
    loss = 0
    sum_true = 0
    for ii in range(data_n):
        x_tmp = train_x[ii].T
        y_tmp = train_y[ii].T
        hat_y = feedforward(weight, bias, tmp_value, x_tmp)
        backpropagation(weight, bias, tmp_value, y_tmp, x_tmp, para_eta=learning_rate)
        loss += loss_func(hat_y, y_tmp)
        sum_true += is_true(hat_y, y_tmp)
    if epoch_i % 20 == 0:
        print('epoch_i :', epoch_i)
        print('loss :', loss)
        print('average accuracy :', sum_true/data_n)

print('predict all data: ')
hat_test = np.mat(np.ones([len(train_y), output_size]))
for ii in range(len(train_y)):
    hat_test[ii] = feedforward(weight, bias, tmp_value, train_x[ii].T).T
print(accuracy(hat_test, train_y))