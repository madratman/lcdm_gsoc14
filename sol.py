from astropy.io import fits
import numpy as np
import scipy.ndimage as sim
from matplotlib import colors, cm, pyplot as plt


m86HDU = fits.open('M86.fits')
m86data = m86HDU[0].data
# print m86data

threshold = m86data.mean() + m86data.std()

labels = sim.label(m86data > threshold)
nplabels = labels[0]

truedata = m86data*nplabels
# print truedata
# print nplabels
# print labels
# print len(labels)
# com =  sim.center_of_mass(m86data)
com_light = sim.center_of_mass(m86data, nplabels)
# print com
# print com_light

# plt.plot(nplabels)
# plt.plot(m86data)
# plt.show()

def main():
    xbar, ybar, cov = inertial_axis(truedata)
    fig, ax = plt.subplots()
    norm = colors.LogNorm(m86data.mean() + 0.5 * m86data.std(), m86data.max(), clip='True')
    plt.imshow(m86data, cmap=cm.gray, norm=norm, origin="lower")
    # ax.imshow(truedata)
    plot_bars(xbar, ybar, cov, ax)
    plt.show()


def rawMoment(truedata, iord, jord):
    nrows, ncols = truedata.shape
    y, x = np.mgrid[:nrows, :ncols]
    # print y, x
    truedata = truedata * x**iord * y**jord
    # print truedata.sum()
    return truedata.sum()

def inertial_axis(truedata):
    # Calculate the x-mean, y-mean, and cov matrix of an image.
    truedata_sum = truedata.sum()
    mx = rawMoment(truedata, 1, 0)
    my = rawMoment(truedata, 0, 1)
    # print mx, my
    x_bar = mx / truedata_sum
    y_bar = my / truedata_sum
    # print x_bar, y_bar
    u11 = (rawMoment(truedata, 1, 1) - x_bar * my) / truedata_sum
    u20 = (rawMoment(truedata, 2, 0) - x_bar * mx) / truedata_sum
    u02 = (rawMoment(truedata, 0, 2) - y_bar * my) / truedata_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov

def plot_bars(x_bar, y_bar, cov, ax):
    def make_lines(eigvals, eigvecs, mean, i):
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color='yellow')
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), marker='o', color='blue')
    ax.axis('image')



if __name__ == '__main__':
	main()    
    
