from astropy.io import fits
import numpy as np
import scipy.ndimage as sim
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


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
# plt.plot([1,2,3,4], [1,4,9,16])
# plt.show()

def main():
    
    xbar, ybar, cov = inertial_axis(truedata)


    # norm=lm(truedata.mean() + 0.5 * truedata.std(), truedata.max(), clip='True'), \
    # cmap = cm.gray,origin="lower") 
    fig, ax = plt.subplots()
    # ax.imshow(truedata,cmap = cm.gray,LogNorm(truedata.mean() + 0.5 * truedata.std(), truedata.max())
    plot_bars(xbar, ybar, cov, ax)
    plt.show()


def rawMoment(m86data, iord, jord):
    nrows, ncols = m86data.shape
    y, x = np.mgrid[:nrows, :ncols]
    # print y, x
    m86data = m86data * x**iord * y**jord
    # print m86data.sum()
    return m86data.sum()

def inertial_axis(m86data):
    # Calculate the x-mean, y-mean, and cov matrix of an image.
    m86data_sum = m86data.sum()
    mx = rawMoment(m86data, 1, 0)
    my = rawMoment(m86data, 0, 1)
    print mx, my
    x_bar = mx / m86data_sum
    y_bar = my / m86data_sum
    print x_bar, y_bar
    u11 = (rawMoment(m86data, 1, 1) - x_bar * my) / m86data_sum
    u20 = (rawMoment(m86data, 2, 0) - x_bar * mx) / m86data_sum
    u02 = (rawMoment(m86data, 0, 2) - y_bar * my) / m86data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return x_bar, y_bar, cov

def plot_bars(x_bar, y_bar, cov, ax):
    """Plot bars with a length of 2 stddev along the principal axes."""
    def make_lines(eigvals, eigvecs, mean, i):
        """Make lines a length of 2 stddev."""
        std = np.sqrt(eigvals[i])
        vec = 2 * std * eigvecs[:,i] / np.hypot(*eigvecs[:,i])
        x, y = np.vstack((mean-vec, mean, mean+vec)).T
        return x, y
    mean = np.array([x_bar, y_bar])
    eigvals, eigvecs = np.linalg.eigh(cov)
    ax.plot(*make_lines(eigvals, eigvecs, mean, 0), marker='o', color='blue')
    ax.plot(*make_lines(eigvals, eigvecs, mean, -1), marker='o', color='yellow')
    ax.axis('image')



if __name__ == '__main__':
	main()    
    
