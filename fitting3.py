import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib import cm


def main():
    # Generate Data...
    numdata = 10
    data = getData()
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]

    x = np.random.random(numdata)
    y = np.random.random(numdata)
    z = np.random.uniform(low=-100, high=0, size=(numdata,))

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z)

    # Evaluate it on a grid...
    nx, ny = 50, 50
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    # plt.contour(xx, yy, zz, 50, extent=(x.min(), x.max(), y.min(), y.max()), alpha=0.6)
    # plt.imshow(zz, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', alpha=0.5)
    plt.scatter(x, y, c=z)

    coords = combine(xx, yy, zz)
    x_coords = [item[0] for item in coords]
    y_coords = [item[1] for item in coords]
    z_coords = [item[2] for item in coords]
    plt.scatter(x_coords, y_coords, c=flatten(z_coords), alpha=0.2, cmap='Greys')

    plt.show()
    print('done')

def flatten(x):
    m = interp1d([min(x), max(x)], [1, 0])
    for i, v in enumerate(x):
        x[i] = m(v)

    return x

def combine(xx, yy, zz):
    coords = []
    for i,row in enumerate(xx):
        for j, _ in enumerate(row):
            x = xx[i][j]
            y = yy[i][j]
            z = zz[i][j]
            coords.append([x, y, z])
            print('{}, {}, {}'.format(x, y, z))

    return coords

def getData():

    return np.array([
        [12.5,70,81.32], [25,70,88.54], [37.5,70,67.58], [50,70,55.32],
        [62.5,70,56.84], [77,70,49.52], [0,11.5,71.32], [77,57.5,67.20],
        [0,23,58.54], [25,46,51.32], [37.5,46,49.52], [0,34.5,63.22],
        [25,34.5,48.32], [37.5,34.5,82.30], [50,34.5,56.42], [77,34.5,48.32],
        [37.5,23,67.32], [0,46,64.20], [77,11.5,41.89], [77,46,55.54],
        [77,23,52.22], [0,57.5,93.72], [0,70,98.20], [77,0,42.32]
    ])

def polyfit2d(x, y, z, order=3):
    ncols = (order + 1)**2
    G = np.zeros((x.size, ncols))
    ij = itertools.product(range(order+1), range(order+1))
    for k, (i,j) in enumerate(ij):
        G[:,k] = x**i * y**j
    m, _, _, _ = np.linalg.lstsq(G, z, rcond=None)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

main()