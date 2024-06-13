'''Analysis facilities for the performance evaluation of the reconstruction
of both NN and standard reco method.
'''

import numpy as np
from scipy.optimize import curve_fit

def doublegaussian(x: float, N1: float, mu1: float, sigma1: float, N2: float,
                   mu2: float, sigma2: float) -> float :
    '''Defining the sum of two Gaussians
    '''
    return N1*(np.exp(-((x-mu1)**2)/(2*sigma1**2))) + N2*(np.exp(-((x-mu2)**2)/(2*sigma2**2)))

def resolution_evaluation():
