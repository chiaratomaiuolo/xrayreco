"""Evaluation of NN performance on a test set and comparison with the standard
reconstruction strategy based on nearest-neighbors for the pixel track selection.
The reconstruction strategy computes:
- The sum over pixels (after zero suppression) as energy estimation;
- The barycenter as hit coordinate estimation.
"""

from pathlib import Path
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np

from xrayreco.fitfacilities import fit_doublegauss
from xrayreco.dataprocessing import Xraydata, processing_training_data, recon_data,\
                                   highest_pixel_coordinates
from xrayreco.nnmodels import DNN_e, DNN_xy


# Root folder of the package
XRAYRECO_ROOT = (Path(__file__).parent).parent
# Folder containing the datasets
XRAYRECO_DATA = XRAYRECO_ROOT / 'datasets'
# Folder containing training weights and scaler
XRAYRECO_TRAINING = XRAYRECO_ROOT / 'training'

#Defining the argument parser and its entries
parser = argparse.ArgumentParser(description='Testing of the neural networks\
                                 for the reconstruction of energy and position\
                                 of incident X-rays on the detector.')
parser.add_argument('--encheckpointpath', type=str, 
                    default='training/cp_e_fulldata.weights.h5', help='Path to the\
                    checkpoint file for the energy NN')
parser.add_argument('--xycheckpointpath', type=str, 
                    default='training/cp_xy_fulldata.weights.h5', help='Path to the\
                    checkpoint file for the hit position NN')


if __name__ == "__main__":
    # Parsing the argument parser arguments
    args = parser.parse_args()
    # Loading the test data file, both raw and reconstructed
    # The test datafile has 20 ENC electronic noise for each pixel and 
    # the source is a Cu one.
    # Extracting raw file ...
    test_data = Xraydata(XRAYRECO_DATA / 'hxsim_20ENC_01srcsigma_test.h5')
    # ... and extracting recon data columns
    recon_e, recon_x, recon_y = recon_data(XRAYRECO_DATA / 'hxsim_20ENC_01srcsigma_test_recon.h5')
    # Extracting the highest pixel coordinates for reconstructed position rescaling
    # and performing the rescaling.
    x_max, y_max = highest_pixel_coordinates(test_data)
    recon_x = x_max - recon_x
    recon_y = y_max - recon_y

    #Preprocessing test data for obtaining input and target ones
    test_input_data, test_target_data = processing_training_data(test_data)
    # Loading training scaler and applying it on test data
    scaler = joblib.load(XRAYRECO_TRAINING / 'scaler.gz')
    X = scaler.fit_transform(test_input_data.reshape(-1, test_input_data.shape[-1]))\
        .reshape(test_input_data.shape)

    # Compiling the models and loading the weight checkpoints
    model_e = DNN_e(path_to_weights = Path(args.encheckpointpath))
    model_xy = DNN_xy(path_to_weights = Path(args.xycheckpointpath))

    # Predicting energy and position of test file
    predicted_energies = model_e.predict(X)
    predicted_xy = model_xy.predict(X)

    content, bins = np.histogram(predicted_energies, bins=50)
    
    errors = np.sqrt(content) # errors on hist
    errors[errors==0] = 1

    # Fitting the energy for both predicted and reconstructed histograms
    plt.figure()
    fit_doublegauss(recon_e, label='Reconstructed energy')
    fit_doublegauss(predicted_energies, label='NN prediction')
    plt.annotate(r'$E_{K\alpha} = 8046$ eV',
            xy=(8000, 4000), xycoords='data',
            xytext=(-120, 0), textcoords='offset points',
            size=18, va='center',
            bbox=dict(boxstyle='round', fc='1', ec='k'))
    plt.axvline(8046, linewidth=3, color='k')
    plt.annotate(r'$E_{K\beta} = 8906$ eV',
            xy=(9000, 1000), xycoords='data',
            xytext=(0, 0), textcoords='offset points',
            size=18, va='center',
            bbox=dict(boxstyle='round', fc='1', ec='k'))
    plt.axvline(8906, linewidth=3, color='k')
    plt.legend()

    # Evaluating the position stats:
    # computing residuals with respect to the MC truth for both recon and predicted
    # Reconstructed residuals wrt MC truth
    recon_res_x = recon_x-test_target_data[:,1]
    recon_res_y = recon_y-test_target_data[:,2]
    # NN predictions residuals wrt MC truth
    pred_res_x = predicted_xy[:,0]-test_target_data[:,1]
    pred_res_y = predicted_xy[:,1]-test_target_data[:,2]
    # Plotting the distributions
    f, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.hist(recon_res_x, bins=50, label='Recon x - MC x \n' +
             rf'$\mu = {np.mean(recon_res_x):.3f}$' + '\n' +
             rf'$\sigma = {np.std(recon_res_x):.3f}$')
    ax1.hist(pred_res_x, bins=50, label='Predicted x - MC x \n' +
             rf'$\mu = {np.mean(pred_res_x):.3f}$' + '\n' +
             rf'$\sigma = {np.std(pred_res_x):.3f}$')
    ax1.set(xlabel=r'$x-x_{MC}$ [cm]', ylabel='Counts')
    ax1.legend()
    ax2.hist(recon_res_y, bins=50, label='Recon y - MC y \n' +
             rf'$\mu = {np.mean(recon_res_y):.3f}$' + '\n' +
             rf'$\sigma = {np.std(recon_res_y):.3f}$')
    ax2.hist(pred_res_y, bins=50, label='Predicted x - MC x \n' +
             rf'$\mu = {np.mean(pred_res_y):.3f}$' + '\n' +
             rf'$\sigma = {np.std(pred_res_y):.3f}$')
    ax2.set(xlabel=r'$y-y_{MC}$ [cm]', ylabel='Counts')
    ax2.legend()
    

    # Closing files
    del test_data

    plt.show()











