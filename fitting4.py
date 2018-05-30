from numpy.polynomial import polynomial
import numpy as np


def polyfit2d(x, y, f, deg):
    x = np.asarray(x)
    y = np.asarray(y)
    f = np.asarray(f)
    deg = np.asarray(deg)
    vander = polynomial.polyvander2d(x, y, deg)
    vander = vander.reshape((-1, vander.shape[-1]))
    f = f.reshape((vander.shape[0],))
    c = np.linalg.lstsq(vander, f)[0]
    return c.reshape(deg + 1)


def main(x, y, c, deg):
    X = np.array(np.meshgrid(x, y))
    f = polynomial.polyval2d(X[0], X[1], c)
    c1 = polyfit2d(X[0], X[1], f, deg)


main([-1, 2, 3],
     [4, 5],
     [[1, 2], [4, 5], [0, 0]],
     [1, 1])
