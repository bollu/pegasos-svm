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
# ts = training samples. List of (xi, yi)
# Fig 1. Pegasos algorithm
def train_linear(d, lam, ts, debug=False):
    T = len(ts)
    w = np.zeros(d)
    t = 1
    while t <= T:
        eta = 1.0 / (float(lam) * float(t))
        (x, y) = urand(ts)
        if y * dot(w, x) < 1:
            w = (1 - eta * lam) * w + eta * y * x
        else: 
            w = (1 - eta * lam) * w

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
    return 1 if dot(w, x) >= 1 else -1

# create a gaussian kernel for d dimensions. Use
# the outer function to create a closure for the inner function.
@numba.jit(nopython=True)
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
    return (s + 1) ^ p

# Figure 3: kernelized pegasos
# d: dimension
# lambda: tuning parameter
# ts: training samples
# K: kernel function:  (training vec x training vec -> float)
@numba.jit(nopython=True)
def train_kernel_gauss(lam, ts, debug=False):
    T = len(ts)
    lam = float(lam)
    n = len(ts)
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
            print("\rdone: " , float(t)/T * 100.0, "%")
    return a


# @numba.jit(nopython=True)
def train_kernel_poly(lam, ts, pow=3, debug=False):
    T = len(ts)
    lam = float(lam)
    n = len(ts)
    # alpha
    a = np.zeros(T)
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

        s *= yi * 1.0 / (lam * float(t))

        if s < 1:
            a[i] = a[i] + 1
        t += 1
        print("\rdone: " , float(t)/T * 100.0, "%")
    return a

# specialization of classify_kernel for K = polynomialK
# @numba.jit(nopython=True)
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
    

def train_test_quad_kernel():
    print("QUADRATIC (with kernel):")
    NTRAIN = 1000
    ts = []
    for _ in range(NTRAIN):
        x1 = np.random.rand() * 10
        x2 = np.random.rand() * 10
        t = bool2y(x1 > 3 *  x2 * x2)
        ts.append((np.asarray([x1, x2]), t))

    a = train_kernel_poly(0.01, ts, debug=False)

    loss = 0
    ts = []
    NTEST = 10
    for _ in range(NTEST):
        x1 = np.random.rand()
        x2 = np.random.rand()
        t = bool2y(x1 > 3 * x2)
        if classify_kernel(a, np.asarray([x1, x2]), ts, gaussianK) != t:
            loss += 1 

    print("total loss: ", loss)
    print("avg loss: ", loss / NTEST)


# return list of tuples of (x, y)
def load_mnist(path, kind):
    (xs, ys) = mnist_reader.load_mnist(path, kind=kind)
    legal = ys <= 1
    xs = xs[legal]
    ys = ys[legal]
    return list(zip(xs, ys))

# train the fashion kernel on the large dataset
def train_fashion_kernel(N):
    print("FASHION (first %s): " % N)
    ts = load_mnist(".",kind="train")
    ts = ts[:N]
    print("fashion dataset sample: ", ts[0])

    a = train_kernel_poly(0.01, ts, debug=False)
    return a



def test_fashion_kernel(a):
    # take the first 'a' lenght sample from the traning set
    # since the vector 'a' describes their weights
    ts = load_mnist(".", kind="train")[:len(a)]

    tests = load_mnist(".", kind="t10k")
    tests = tests[:1000]

    print("number of test samples: ", len(tests))
    loss = 0
    for (x,y) in tests:
        if classify_kernel_poly(a, x, ts) != y:
            loss += 1
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
    trainfashion = sub.add_parser("trainfashion", help="train the model")
    testfashion = sub.add_parser("testfashion", help="test the fashion")
    dummyfashion = sub.add_parser("dummyfashion", help="test the model")

    return p.parse_args(s)

if __name__ == "__main__":
    
    p = parse(sys.argv[1:])
    if p.command == "trainfashion":
        a = train_fashion_kernel(10000)
        with open("kernel-coeff.bin", "wb") as f:
            pickle.dump(a, f)
    if p.command == "testfashion":
        with open("kernel-coeff.bin", "r") as f:
            a = pickle.load(f)
        test_fashion_kernel(a)
    else:
        a = train_fashion_kernel(1000)
        test_fashion_kernel(a)
    pass
    

