#!/usr/bin/env python3
import math
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
import mnist_reader
from itertools import *
import numba
import pickle
import sys
import argparse


class ETA:
    def __init__(self, _n):
        self.n = _n
        self.i = 0

    def bump(self):
        self.i += 1

    def ratio(self):
        return float(self.i) / self.n

    def percent(self):
        return "%4.2f %%" % (self.ratio() * 100)

# d = dimension
# lam=lambda
# ts = training samples. List of (xi, yi)
# T = number of iterations
# Fig 1. Pegasos algorithm
def train_linear(d, lam, ts, T=None, debug=False):
    if T is None:
        T = len(ts) * 2

    w = np.zeros(d) # weight vector
    t = 1 # current iteration
    ixs = randint(0, len(ts), size=T + 1) # generate random indeces

    # loop for the samples
    while t <= T:
        # calculate eta
        eta = 1.0 / (float(lam) * float(t))
        # current sample (x, y)
        (x, y) = ts[ixs[t]]

        if y * np.dot(w, x) < 1:
            w = (1 - eta * lam) * w + eta * y * x
        else: 
            w = (1 - eta * lam) * w

        # Debugging code to plot the hyperplanes
        if debug and t % (T//50) == 0:
            if d == 2:
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                pxs = []
                pys = []
                pcs = []
                for ((x1, x2), y) in ts[:1000]:
                    if x1 * w[0] + x2 * w[1] >= 1:
                        pxs.append(x1)
                        pys.append(x2)
                        pcs.append('red' if y == 1 else 'blue')
                ax.scatter(pxs, pys, c=pcs, marker='v')

                pxs = []
                pys = []
                pcs = []
                for ((x1, x2), y) in ts[:1000]:
                    if x1 * w[0] + x2 * w[1] <= 1:
                        pxs.append(x1)
                        pys.append(x2)
                        pcs.append('red' if y == 1 else 'blue')
                ax.scatter(pxs, pys, c=pcs, marker='+')

                # hyperplane is perpendicular to w
                # so points will be (x1, x2) such that (x1w1 + x2w2 >= 1)
                # x0w0 + x1w1 >= 1
                # x1 >= (1 - x0w0) / w1 = 1/w1 - x0 w0/w1
                # x2 >= 1/w1 - x0 * w0 / w1
                xx = np.linspace(-10, 10)
                yy = 1/w[1] - xx * w[0] / w[1]
                ax.plot(xx, yy)

                plt.show()

        t += 1
    return w

def classify_linear(w, x):
    return 1 if np.dot(w, x) >= 1 else -1

# create a gaussian kernel for d dimensions. Use
# the outer function to create a closure for the inner function.
def gaussianK(x1, x2):
    v = x1 - x2
    nsq = 0.0
    for i in range(len(x1)):
        nsq += v[i] * v[i]
    sigma = 1.0
    return math.e ** (-nsq / (2 * sigma))

# @numba.jit(nopython=True)
# d: dimension
# p: power to raise
def polynomialK(x1, x2, d, p):
    s = np.dot(x1, x2)
#    s = 0
#    i = 0
#    while i < d:
#        s += x1[i] * x2[i]
#        i += 1
#
    return (s + 1) ** p

# Figure 3: kernelized pegasos
# d: dimension
# lambda: tuning parameter
# ts: training samples
# K: kernel function:  (training vec x training vec -> float)
def train_kernel_gauss(lam, ts, debug=False):
    T = len(ts)
    lam = float(lam)
    # alpha
    a = np.zeros(T)
    t = 1
    ixs = randint(0, n, size=(T+2))
    print(" ")
    while t <= T:
        i = ixs[t]
        (xi, yi) = ts[i]

        # score
        s = 0
        j = 0
        while j < n:
            (xj, _) = ts[j]
            s += a[j] * yi * gaussianK(xi, xj)
            j += 1

        s *= yi * 1.0 / (lam * float(t))

        if s < 1:
            a[i] = a[i] + 1
        t += 1
        if t % (T // 100) == 0:
            print("\rtraining: %4.2f %%" %(float(t)/T * 100.0), end='')
    return a


# Figure 3
# lam = lambda
# ts = training samples. List of (xi, yi)
# T = number of timesteps
# pow = polynomial to raise the kernel: (1 + x_i . x_j)^pow
def train_kernel_poly(lam, ts, T=None, pow=3):
    if T is None:
        T = len(ts)*3

    print ("training poly kernel. #samples: %d | lambda: %4.3f | T: %d | pow: %4.2f" % 
                (len(ts), lam, T, pow))
    lam = float(lam)
    n = len(ts)
    # alpha
    a = np.zeros(len(ts))
    # dimension of the training vectors
    d = len(ts[0][0])
    # training sample
    t = 1
    ixs = randint(0, n, size=(T+2))
    print(" ")
    while t <= T:
        i = ixs[t]
        (xi, yi) = ts[i]

        # score
        s = 0
        j = 0
        while j < n:
            (xj, _) = ts[j]
            s += a[j] * yi * polynomialK(xi, xj, d, pow)
            j += 1

        s *=  yi * 1.0 / (lam * float(t))

        if s < 1:
            a[i] = a[i] + 1
        t += 1
        if t % (T // 100) == 0:
            print("\rtraining: %4.2f %%" %(float(t)/T * 100.0), end='')
    return a

# specialization of classify_kernel for K = polynomialK
def classify_kernel_poly(a, x, ts, pow=3):
    d = ts[0][0]
    s = 0
    for j in range(len(ts)):
        (xj, _) = ts[j]
        s += a[j] * polynomialK(x, xj, d, pow)
    return 1 if s >= 1 else -1

# a: alpha computed from training
# x: point to classify
# ts: training samples
# K: kernel function
def classify_kernel(a, x, ts, K):
    s = 0
    for j in range(len(ts)):
        (xj, _) = ts[j]
        s += a[j] * K(x, xj)

    return 1 if s >= 1 else -1

def bool2y(b):
    return 1 if b else -1



def train_test_linreg_linear():
    print("LINEAR REGRESSION (with linear SVM): ")
    NTRAIN = 100000
    ts = []
    for _ in range(NTRAIN):
        x1 = np.random.rand() * 10
        x2 = np.random.rand() * 10
        t = bool2y(x1 > 3 *  x2)
        ts.append((np.asarray([x1, x2]), t))

    w = train_linear(2, 0.01, ts, debug=False)

    loss = 0
    ts = []
    NTEST = 1000
    for _ in range(NTEST):
        x1 = np.random.rand()
        x2 = np.random.rand()
        t = bool2y(x1 > 3 * x2)
        if classify_linear(w, np.asarray([x1, x2])) != t:
            loss += 1 

    print("total loss: ", loss)
    print("avg loss: ", loss / NTEST)
    

def train_test_cube_kernel():
    NTRAIN = 500
    print("CUBIC with kernel (#training: %d):" % NTRAIN)
    ts = []
    for _ in range(NTRAIN):
        x1 = np.random.rand() * 10
        x2 = np.random.rand() * 10
        t = bool2y(x1 >=  x2 * x2  * x2)
        ts.append((np.asarray([x1, x2]), t))

    a = train_kernel_poly(0.01, ts, T=len(ts)*10)

    loss = 0
    NTEST = 100
    print("\n\nrunning tests (total %d)" % NTEST)
    for i in range(NTEST):
        x1 = np.random.rand()
        x2 = np.random.rand()
        t = bool2y(x1 >=  x2 * x2 * x2)
        print("\rtesting: %4.2f %%" %(100.0 * i / NTEST), end='')
        if classify_kernel_poly(a, np.asarray([x1, x2]), ts) != t:
            loss += 1 
    print(" ")
    print("total loss: ", loss)
    print("avg loss: ", loss / NTEST)


# return list of tuples of (x, y)
def load_mnist(path, kind):
    (xs, ys) = mnist_reader.load_mnist(path, kind=kind)

    ts = []
    for i in range(len(xs)):
        if ys[i] > 1: continue
        y = 1 if ys[i] == 0 else -1
        ts.append((xs[i], y))

    return ts

# train the fashion kernel on the large dataset
def train_fashion_kernel(N):
    print("FASHION (first %s): " % N)
    ts = load_mnist(".",kind="train")
    ts = ts[:N]
    print("fashion dataset sample: ", ts[0])

    a = train_kernel_poly(0.01, ts, T=len(ts)*10)
    return a



def test_fashion_kernel(a):
    # take the first 'a' lenght sample from the traning set
    # since the vector 'a' describes their weights
    ts = load_mnist(".", kind="train")[:len(a)]

    tests = load_mnist(".", kind="t10k")
    tests = tests[:300]

    print("\n\n")
    print("#train : ", len(a))
    print("#test samples: ", len(tests))
    loss = 0
    i = 0
    N = len(tests)
    for (x,y) in tests:
        if classify_kernel_poly(a, x, ts) != y:
            loss += 1
        i += 1
        if i % (N // 100) == 0:
            print("\rtesting: %4.2f %%" % (100.0 * float(i)/N), end='')
    print("\n")
    print("total loss: ", loss)
    print("avg loss: ", loss / len(tests))

def sample_train_test_fashion_kernel():
    print("FASHION: ")
    (x_train, y_train) = mnist_reader.load_mnist(".", kind="train")
    ts = list(zip(x_train, y_train))
    ts = ts[:1000]
    print("fashion dataset sample: ", ts[0])


def parse(s):
    p = argparse.ArgumentParser()

    sub = p.add_subparsers(dest="command")
    sub.add_parser("trainfashion", help="Train fashion model and save data")
    sub.add_parser("testfashion", help="Test the fashion model from saved data")
    sub.add_parser("demofashion", help="Train and test the fashion model on a small portion of the dataset")
    sub.add_parser("demolinear", help="Train & test a linear model y = ax")
    sub.add_parser("democubic", help="Train a test a cubic model y = x^3")

    return p.parse_args(s)

if __name__ == "__main__":
    
    p = parse(sys.argv[1:])
    if p.command == "trainfashion":
        a = train_fashion_kernel(10000)
        with open("kernel-coeff.bin", "wb") as f:
            pickle.dump(a, f)
    elif p.command == "testfashion":
        with open("kernel-coeff.bin", "rb") as f:
            a = pickle.load(f)
        test_fashion_kernel(a)
    elif p.command =="demofashion":
        a = train_fashion_kernel(400)
        test_fashion_kernel(a)
    elif p.command =="democubic":
        train_test_cube_kernel()
    elif p.command =="demolinear":
        train_test_linreg_linear()
    else:
        print("please invoke with option. See --help for all options")
    

