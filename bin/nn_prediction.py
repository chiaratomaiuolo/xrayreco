"""NN prediction script.
"""

from pathlib import Path
import argparse

import joblib
import matplotlib.pyplot as plt
import numpy as np

from hexsample.fileio import ReconInputFile
from xrayreco.preprocessing import Xraydata, processing_data, recon_data,\
                                   highest_pixel_coordinates
from xrayreco.nnmodels import DNN_e, DNN_xy

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
parser.add_argument('rawdatafile', type=str, help='Path to the raw data file\
                    in .h5 format containing the event tracks to be predicted')
parser.add_argument('--encheckpointpath', type=str, 
                    default='cp_e_fulldata.weights.h5', help='Name of the\
                    checkpoint file for the energy NN to be found in the training\
                    directory')
parser.add_argument('--xycheckpointpath', type=str, 
                    default='cp_xy_fulldata.weights.h5', help='Name of the\
                    checkpoint file for the position NN to be found in the training\
                    directory')

if __name__ == '__main__':
    # Parsing the argument parser
    args = parser.parse_args()

    # Defining the checkpoint paths ...
    energy_checkpoint_path = XRAYRECO_TRAINING / args.encheckpointpath
    xy_checkpoint_path = XRAYRECO_TRAINING / args.xycheckpointpath
    
    # ... compiling the models and loading the weights from checkpoints.
    model_e = DNN_e(path_to_weights=energy_checkpoint_path)
    model_xy = DNN_xy(path_to_weights=xy_checkpoint_path)

    # Loading training scaler 
    scaler = joblib.load(XRAYRECO_TRAINING / 'scaler.gz')

    # Opening a datafile...
    raw_data = Xraydata(args.rawdatafile)
    # ... preprocessing it ...
    input_data = processing_data(raw_data)
    # ... rescaling it ...
    X = scaler.fit_transform(input_data.reshape(-1, input_data.shape[-1]))\
        .reshape(input_data.shape)
    
    # ... finally performing the prediction
    predicted_e = model_e.fit(X)
    predicted_xy = model_xy.fit(X)