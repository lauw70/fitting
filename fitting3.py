import itertools
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm


def main():
    # Generate Data...
    numdata = 500
    data = getData()
    x = data[:,0]
    y = data[:,1]
    z = data[:,2]
    #y = [item[1] for item in data]
    #z = [item[2] for item in data]
    #x = np.random.random(numdata)
    #y = np.random.random(numdata)
    #z = x**2 + y**2 + 3*x**3 + y + np.random.random(numdata)

    # Fit a 3rd order, 2d polynomial
    m = polyfit2d(x,y,z, order=3)

    # Evaluate it on a grid...
    nx, ny = 50, 50
    xx, yy = np.meshgrid(np.linspace(x.min(), x.max(), nx),
                         np.linspace(y.min(), y.max(), ny))
    zz = polyval2d(xx, yy, m)

    # Plot surface
    fig = plt.figure()
    ax = fig.add_subplot(121)
    # ax.contourf(xx, yy, zz, 100)
    ax.scatter(xx, yy, c=zz)
    ax.scatter(x, y, c=z)


    # Plot
    fig = plt.figure()
    ax2 = fig.add_subplot(122)
    ax2.imshow(zz, extent=(x.min(), y.max(), x.max(), y.min()))
    ax2.scatter(x, y, c=z)
    plt.show()

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
    m, _, _, _ = np.linalg.lstsq(G, z)
    return m

def polyval2d(x, y, m):
    order = int(np.sqrt(len(m))) - 1
    ij = itertools.product(range(order+1), range(order+1))
    z = np.zeros_like(x)
    for a, (i,j) in zip(m, ij):
        z += a * x**i * y**j
    return z

main()