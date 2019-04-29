#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import *
import mnist_reader

def z(n): return np.zeros(n)
def urand(xs): return xs[randint(0, len(xs))]
def dot(x, y): return np.dot(x, y)
# d = dimension
# ts = training samples. List of (xi, yi)
def train_linear(d, lam, T, ts, debug=False):
    w = z(d)
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

def classify(w, x):
    return 1 if dot(w, x) >= 1 else -1

def bool2y(b):
    return 1 if b else -1


def train_test_linear():
    NTRAIN = 100000
    ts = []
    for _ in range(NTRAIN):
        x1 = np.random.rand() * 10
        x2 = np.random.rand() * 10
        t = bool2y(x1 > 3 *  x2)
        ts.append((np.asarray([x1, x2]), t))

    w = train_linear(2, 0.025, len(ts), ts, debug=False)

    loss = 0
    ts = []
    NTEST = 1000
    for _ in range(NTEST):
        x1 = np.random.rand()
        x2 = np.random.rand()
        t = bool2y(x1 > 3 * x2)
        if classify(w, np.asarray([x1, x2])) != t:
            loss += 1 

    print("total loss: ", loss)
    print("avg loss: ", loss / NTEST)
    

if __name__ == "__main__":
    train_test_linear()
    pass
    

