'''Training of the NN for the reconstruction of the energy and
the hit coordinates of X-rays impinging the detector.
'''

from pathlib import Path
import argparse

import joblib
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from xrayreco.preprocessing import Xraydata, processing_data
from xrayreco.nnmodels import DNN_e, DNN_xy
from hexsample.fileio import ReconInputFile

# Root folder of the package
XRAYRECO_ROOT = (Path(__file__).parent).parent
# Folder containing the datasets
XRAYRECO_DATA = XRAYRECO_ROOT / 'datasets'
# Folder containing training weights and scaler
XRAYRECO_TRAINING = XRAYRECO_ROOT / 'training'

#Defining the argument parser and its entries
parser = argparse.ArgumentParser(description='Training of the neural networks\
                                 for the reconstruction of energy and position\
                                 of incident X-rays on the detector.')
parser.add_argument('--enweights', type=str, default=None, help='Path to a\
                    .weghts.h5 file for loading already-present weights in the\
                    NN for energy regression')
parser.add_argument('--xyweights', type=str, default=None, help='Path to a\
                    .weghts.h5 file for loading already-present weights in the\
                    NN for hit position regression')
parser.add_argument('--encheckpointpath', type=str, 
                    default='cp_e_fulldata.weights.h5', help='Path to the\
                    checkpoint file for the energy NN')
parser.add_argument('--xycheckpointpath', type=str, 
                    default='cp_xy_fulldata.weights.h5', help='Path to the\
                    checkpoint file for the hit position NN')

if __name__ == "__main__":
    full_input_dataset = []
    full_output_dataset = []
    # Collecting the arguments from the argument parser
    args = parser.parse_args()
    # Storing the datasets. Using 3 simulations, with 0,20,40 ENC for single px
    data_0ENC = Xraydata(XRAYRECO_DATA / 'hxsim_0ENC_01srcsigma_continuumspectrum.h5')
    data_20ENC = Xraydata(XRAYRECO_DATA / 'hxsim_20ENC_01srcsigma_continuumspectrum.h5')
    data_40ENC = Xraydata(XRAYRECO_DATA / 'hxsim_40ENC_01srcsigma_continuumspectrum.h5')

    datasets = [data_0ENC, data_20ENC, data_40ENC]

    for data in datasets:
        input_data, output_data = processing_data(data)
        full_input_dataset.append(input_data)
        full_output_dataset.append(output_data)
    
    full_input_dataset = np.concatenate(full_input_dataset, axis=0)
    full_output_dataset = np.concatenate(full_output_dataset, axis=0)

    # Standardizing input data
    # This is a pt where I can think a parallelization for large datasets
    scaler = StandardScaler()
    # Saving the scaler for using it for test set and evaluations
    joblib.dump(scaler, XRAYRECO_TRAINING / 'scaler.gz')
    X_train = scaler.fit_transform(full_input_dataset.reshape(-1, full_input_dataset.shape[-1])).reshape(full_input_dataset.shape)
    
    #Dividing the energy and hit position target data (just for clarity)
    target_energies = full_output_dataset[:,0]
    target_xy = full_output_dataset[:,1:]

    # Defining the checkpoint paths
    energy_checkpoint_path = XRAYRECO_TRAINING / args.encheckpointpath
    xy_checkpoint_path = XRAYRECO_TRAINING / args.xycheckpointpath
    
    # Defining the two NNs, one for the energy, one for the position.
    model_e = DNN_e(path_to_weights=energy_checkpoint_path)
    model_xy = DNN_xy(path_to_weights=xy_checkpoint_path)
    
    # Creating the new checkpoints
    cp_callback_e = ModelCheckpoint(filepath=energy_checkpoint_path,
                                    save_weights_only=True, verbose=1)
    cp_callback_xy = ModelCheckpoint(filepath=xy_checkpoint_path,
                                     save_weights_only=True, verbose=1)

    # Training the NNs
    history_e  = model_e.fit(X_train, target_energies,
                             validation_split=0.05, epochs=40, callbacks=[cp_callback_e])
    history_xy  = model_xy.fit(X_train, target_xy,
                             validation_split=0.05, epochs=40, callbacks=[cp_callback_xy])
    
    # Closing files 
    for data in datasets:
        del data

    # Plotting the loss trend over epochs for the energy
    plt.figure('Loss over epochs for energy NN')
    plt.plot(history_e.history['val_loss'], label='Validation loss')
    plt.plot(history_e.history['loss'], label='Training loss')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Plotting the loss trend over epochs for the hit position
    plt.figure('Loss over epochs for hit position NN')
    plt.plot(history_xy.history['val_loss'], label='Validation loss')
    plt.plot(history_xy.history['loss'], label='Training loss')
    plt.xlabel('Epoch number')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    #That's all for this script, the comparison with the std reco is done in another script
    
    #Showing the pictures
    plt.show()



