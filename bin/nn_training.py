'''Training of the NN for the reconstruction of the energy and
the hit coordinates of X-rays impinging the detector.
'''

from pathlib import Path
import argparse

import numpy as np

from xrayreco.preprocessing import Xraydata
from xrayreco.nnmodels import DNN_e, DNN_xy
from hexsample.fileio import ReconInputFile

# Root folder of the package
XRAYRECO_ROOT = Path(__file__).parent
# Folder containing the datasets
XRAYRECO_DATA = XRAYRECO_ROOT / 'datasets'

#Defining the argument parser and its entries
parser = argparse.ArgumentParser(description='Training of the neural networks\
                                 for the reconstruction of energy and position\
                                 of incident X-rays on the detector.')
parser.add_argument('enweights', type='str', default=None, help='Path to a\
                    .weghts.h5 file for loading already-present weights in the\
                    NN for energy regression')
parser.add_argument('xyweights', type='str', default=None, help='Path to a\
                    .weghts.h5 file for loading already-present weights in the\
                    NN for hit position regression')
parser.add_argument('encheckpointpath', type='str', 
                    default='training/cp_e.weights.h5', help='Path to the\
                    checkpoint file for the energy NN')
parser.add_argument('xycheckpointpath', type='str', 
                    default='training/cp_xy.weights.h5', help='Path to the\
                    checkpoint file for the hit position NN')

if __name__ == "__main__":
    args = parser.parse_args()
    # Storing the datasets. Using 3 simulations, with 0,20,40 ENC for single px
    data_0ENC = Xraydata(XRAYRECO_DATA / 'hxsim_0ENC.h5')
    data_20ENC = Xraydata(XRAYRECO_DATA / 'hxsim_20ENC.h5')
    data_40ENC = Xraydata(XRAYRECO_DATA / 'hxsim_40ENC.h5')
    # Creating a list in order to loop over datasets
    datasets = [data_0ENC, data_20ENC, data_40ENC]
    full_input_dataset = np.empty(0)
    full_output_dataset = np.empty(0)
    # Storing all input and output data inside a single overall dataset
    for data in datasets:
        full_input_dataset = np.append(full_input_dataset, data.input_events_data(),
                                       axis=0)
        full_output_dataset = np.append(full_output_dataset, data.target_data(),
                                       axis=0)
    
    #Dividing the energy and hit position target data (just for clarity)
    target_energies = full_output_dataset[:,0]
    target_xy = full_output_dataset[:,1:]
    
    # Defining the two NNs, one for the energy, one for the position.
    model_e = DNN_e(path_to_weights=args.enweights)
    model_xy = DNN_xy(path_to_weights=args.xyweights)

    # Training the NNs
    history_e  = model_e.fit(full_input_dataset, target_energies, validation_split=0.05, epochs=50, callbacks=[args.encheckpointpath])
    history_xy  = model_e.fit(full_input_dataset, target_xy, validation_split=0.05, epochs=50, callbacks=[args.xycheckpointpath])






    

