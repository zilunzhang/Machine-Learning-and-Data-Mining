# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
from sklearn.datasets import load_boston
from datetime import datetime

np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
# add constant one feature - no bias needed
x = np.concatenate((np.ones((506, 1)), x), axis=1)
N = x.shape[0] # 506
d = x.shape[1] # 14
# (506, )
y = boston['target']
# produce a set of random number(1-506) and permute them.
idx = np.random.permutation(range(N))


# helper function
def l2(A,B):
    """
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    """
    A_norm = (np.square(A)).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (np.square(B)).sum(axis=1).reshape(1, B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist


# helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    """
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    """
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j, tau in enumerate(taus):
        predictions = np.array([LRLS(x_test[i,:].reshape(d, 1), x_train, y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions-y_test)**2).mean()
    return losses


# to implement
def LRLS(test_datum, x_train, y_train, tau, lam=1e-5):
    """
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    """

    # (X^TxAxX+ lamda * I)xW = X^TxAxY
    # build a empty list contain diagonal of A.
    x_train_N = x_train.shape[0]
    x_train_d = x_train.shape[1]
    A = np.zeros((x_train_N, x_train_N), dtype=float)
    identity_matrix = np.identity(x_train_d)
    content_numerator = -l2(x_train, test_datum.T)
    content_denominator = np.multiply(2, np.power(tau, 2))
    numerator = np.divide(content_numerator, content_denominator)
    log_summation = misc.logsumexp(numerator)
    denominator = np.exp(log_summation)

    for i in range(x_train_N):
        A[i, i] = np.exp(numerator[i])/denominator

    temp = np.matmul(x_train.T, A)
    left_temp = np.matmul(temp, x_train)
    left = left_temp + np.multiply(lam, identity_matrix)
    right = np.dot(temp, y_train)
    w = np.linalg.solve(left, right)
    y_hat = np.matmul(test_datum.T, w)
    # convert to float [[float]] before.
    return float(y_hat)


def normalize_data(x):
    matrix_x_subtract_mean = x - np.mean(x, axis=0, keepdims=True)
    matrix_x_std = np.std(x, axis=0, keepdims=True)
    normal_x = np.divide(matrix_x_subtract_mean, matrix_x_std)
    return normal_x


def run_k_fold(x, y, taus, k):
    """
    Input: x is the N x d design matrix
           y is the N x 1 targets vector
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    """
    loss = []
    # combine x and y, for convenience when doing shuffle.
    concatenate_matrix = np.concatenate((x, y[:, None]), axis=1)
    # random process
    np.random.shuffle(concatenate_matrix)
    print(concatenate_matrix[0, 0])
    # split to k fold
    temp_container = np.array_split(concatenate_matrix, k)
    i = 0
    while i < k:
        print("fold number is:")
        print(i+1)
        # take out the fold.
        fold = np.array(temp_container[i])
        x_test = fold[:, :d]
        print("x_test size is:")
        print(x_test.shape)
        y_test = fold[:, d]
        print("y_test size is:")
        print(y_test.shape)

        x_train = []
        y_train = []
        # deal with other folds.
        for j in range(k):
            if j == i:
                pass
            else:
                other_fold = np.array(temp_container[j])
                print("other_fold shape is:")
                print(other_fold.shape)
                x_train.append(other_fold[:, :d])
                y_train.append(other_fold[:, d])
        i += 1

        x_train = np.concatenate(np.array(x_train))
        y_train = np.concatenate(np.array(y_train))

        print("x train shape:")
        print(x_train.shape)
        print("y train shape:")
        print(y_train.shape)
        print("x test shape:")
        print(x_test.shape)
        print("y test shape:")
        print(y_test.shape)
        temp_loss = run_on_fold(x_test, y_test, x_train, y_train, taus)
        loss.append(temp_loss)
    output = np.array(loss).mean(0)
    print(output.shape)
    print(output)
    return output

if __name__ == "__main__":
    start = datetime.now()
    # In this excersice we fixed lambda (hard coded to 1e-5)
    # and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    # x = normalize_data(x)

    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(losses)
    print("min loss = {}".format(losses.min()))
    plt.ylabel("Loss Value")
    plt.xlabel("r Value")
    print(datetime.now() - start)
    plt.show()
