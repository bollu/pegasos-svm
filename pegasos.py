import numpy as np
from numpy.random import *

def z(n): return np.zeros(n)
def urand(xs): return choice(xs)
def normsq(w): return np.dot(w.T, w)
# d = dimension
# ts = training samples. List of (xi, yi)
def pegasos(d, lam, T, ts):
    w = z(n)
    t = 1
    while t <= T:
        eta = 1.0 / (lam * t)
        (x, y) = urand(ts)
        if y * dot(w, x) < 1:
            w = (1 - eta * lam) * w + eta * y * x
        else: 
            w = (1 - eta * lam) * w
        t += 1
    return w


if __name__ == "__main__":
    pass
    

