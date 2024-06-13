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


if __name__ == "__main__":
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
    
    # Defining the two NNs, one for the energy, one for the position.
    model_e = DNN_e()
    model_xy = DNN_xy()





    

