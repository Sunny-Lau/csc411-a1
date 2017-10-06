# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as misc
from sklearn.datasets import load_boston
np.random.seed(0)

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

#helper function
def run_on_fold(x_test, y_test, x_train, y_train, taus):
    '''
    Input: x_test is the N_test x d design matrix
           y_test is the N_test x 1 targets vector        
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           taus is a vector of tau values to evaluate
    output: losses a vector of average losses one for each tau value
    '''
    N_test = x_test.shape[0]
    losses = np.zeros(taus.shape)
    for j,tau in enumerate(taus):
        predictions =  np.array([LRLS(x_test[i,:].reshape(1,d),x_train,y_train, tau) \
                        for i in range(N_test)])
        losses[j] = ((predictions-y_test) ** 2).mean()
    return losses
 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    
    numerator = np.exp(np.divide(-1 * (l2(test_datum, x_train)), 2 * tau ** 2))
    denominator = np.exp(misc.logsumexp(np.divide(-1 * (l2(test_datum, x_train)), 2 * tau ** 2))) # 

    A = np.divide(numerator, denominator)
    A = np.diag(A[0,:])

    a = np.dot(np.dot(np.transpose(x_train), A),x_train) #X^TAX
    b = np.dot(np.dot(np.transpose(x_train), A),y_train) #X^TAy

    W = np.linalg.solve(a + lam * np.eye(a.shape[0]), b) #solve W

    predict_y = np.dot(test_datum, W)

    return predict_y


def run_k_fold(x,y,taus,k):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           K in the number of folds
    output is losses a vector of k-fold cross validation losses one for each tau value
    '''
    ## TODO

    fold_size = int(np.round(N/k))
    print("The number of dataset is: {} ".format(N))
    print("The fold size is: {} ".format(fold_size))
    print("##############################\n")

    losses = np.empty([k, len(taus)])

    for i in range(k):
        test = idx[i * fold_size:((i + 1) * fold_size)]
        train = [s for s in idx if s not in test] 
        x_test = x[test,:]
        x_train = x[train,:]
        y_test = y[test]
        y_train = y[train]

        print("This is fold # {}: ".format(i))
        print("""   The dimension of X_test: {}, y_test: {}, \n \
                  X_train: {}, y_train: {} \n""".format(x_test.shape, y_test.shape, x_train.shape, y_train.shape))
        print("##############################\n")
        if i < 5:
            print("Calculating fold {} ... ...\n".format(i + 1))

        losses[i,:] = run_on_fold(x_test, y_test, x_train, y_train, taus)

    return np.mean(losses, axis=0)


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0, 3, 200)
    losses = run_k_fold(x, y, taus, k=5)
    plt.plot(taus, losses)

    axe = plt.gca()
    axe.grid()
    axe.set_xlabel('Tau')
    axe.set_ylabel('Average losses')

    plt.show()

    # Try different k value:

    print("min loss = {}".format(losses.min()))

