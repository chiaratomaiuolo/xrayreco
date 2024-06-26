""" Fit models and fit functions for data analysis purpose
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def Gauss(x: np.array, N: float, mean: float, sigma: float) -> float:
    """Gaussian model for fitting purposes
    Arguments
    ---------
    """
    return N*np.exp(-(x-mean)**2/(2*sigma**2))

def DoubleGauss(x: np.array, N1: float, mean1: float, sigma1: float, 
                N2: float, mean2: float, sigma2: float) -> float:
    """Double Gaussian model (sum of two Gaussian models) for fitting purposes.
    """
    return N1*np.exp(-(x-mean1)**2/(2*sigma1**2)) + N2*np.exp(-(x-mean2)**2/(2*sigma2**2))

def fit_doublegauss(x: np.array, bins: int=50, p0: np.array=None, label: str=None):
    """Fits a distribution with a DoubleGaussian model, plots the distribution histogram
    and the fitted curve with a legend box.

    Arguments
    ---------
    x : np.array or array-like
        Array of values of the distribution to be fitted
    bins : int=50
        Number of bins for the histogram to be created
    p0 : np.array or array-like=None
        Array of 6 elements containing the initial parameters of the DoubleGaussian
        for the fit. If None is provided, the initial parameters will be set to:
        [max(content), 8046., 100., max(content[bins[0:-1]>8600]), 8904, 100], where
        content is the array containing the counts of the binned quantity x, the mean
        values for the Gaussian peaks are taken as the Ka and Kb of a Cu source (that
        is the standard dataset used for the evaluation of the NN performance).
    label : str=None
        Label string to be inserted in the plot with the optimal parameters of the fit.

    """
    # Constructing the histogram with the data array
    content, bins = np.histogram(x, bins=bins)

    # Taking bincenters for fitting purpose
    b = (bins[:-1] + bins[1:]) / 2

    # Constructing the histogram errors (Poissonian)
    # Putting 1 on bins without content
    errors = np.sqrt(content) # errors on hist
    errors[errors==0] = 1

    # Constructing the list of initial parameters and fitting
    if p0 is None:
        # If no p0 is provided, the optimal initial params for a Cu source
        # are provided.
        p0 = [max(content), 8046., 100., max(content[bins[0:-1]>8600]), 8904, 100]
        bounds = ([1., 0, 0, 1., 8800, 0],[+np.inf, 8500, 1000, +np.inf, 9300, +np.inf])
        popt, pcov = curve_fit(DoubleGauss, b, content, p0=p0, sigma=errors, bounds=bounds)
    else: 
        popt, pcov = curve_fit(DoubleGauss, b, content, p0=p0, sigma=errors)
    # Exctracting errors on parameters
    pcov_diag = np.sqrt(np.diag(pcov))
    # Plotting the histogram and the optimal parameters
    plt.hist(x, bins=bins)
    w = np.linspace(min(x), max(x), 1000)
    if label is not None:
        label = label + '\n' + rf'$\mu_1$={popt[1]:.0f} $\pm$ {pcov_diag[1]:.0f} eV' +\
                '\n' + rf'$\sigma_1$={popt[2]:.0f} $\pm$ {pcov_diag[2]:.0f} eV' +\
                '\n' + rf'$\mu_2$={popt[4]:.0f} $\pm$ {pcov_diag[4]:.0f} eV' + '\n'\
                + rf'$\sigma_2$={popt[5]:.0f} $\pm$ {pcov_diag[5]:.0f} eV'
        plt.plot(w, DoubleGauss(w, *popt), label=label)
    else:
        label = rf'$\mu_1$={popt[1]:.0f} $\pm$ {pcov_diag[1]:.0f} eV' +\
                '\n' + rf'$\sigma_1$={popt[2]:.0f} $\pm$ {pcov_diag[2]:.0f} eV' +\
                '\n' + rf'$\mu_2$={popt[4]:.0f} $\pm$ {pcov_diag[4]:.0f} eV' + '\n'\
                + rf'$\sigma_2$={popt[5]:.0f} $\pm$ {pcov_diag[5]:.0f} eV'
        plt.plot(w, DoubleGauss(w, *popt), label=label)
    
    plt.xlabel('Photon energy [eV]')
    plt.ylabel('Counts')