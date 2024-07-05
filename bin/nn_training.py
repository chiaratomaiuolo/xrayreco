'''Training of the NN for the reconstruction of the energy and
the hit coordinates of X-rays impinging the detector.
'''

from pathlib import Path
import argparse

import numpy as np
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import multiprocess

from xrayreco.preprocessing import Xraydata, processing_data
from xrayreco.nnmodels import DNN_e, DNN_xy
from hexsample.fileio import ReconInputFile

# Root folder of the package
XRAYRECO_ROOT = (Path(__file__).parent).parent
# Folder containing the datasets
XRAYRECO_DATA = XRAYRECO_ROOT / 'datasets'

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
                    default='training/cp_e.weights.h5', help='Path to the\
                    checkpoint file for the energy NN')
parser.add_argument('--xycheckpointpath', type=str, 
                    default='training/cp_xy.weights.h5', help='Path to the\
                    checkpoint file for the hit position NN')

def process_data_worker(datafile: Xraydata, q1: multiprocess.Queue,
                        q2: multiprocess.Queue):
    input_data, target_data = processing_data(datafile)
    q1.put(input_data)
    #q2.put(target_data)

if __name__ == "__main__":
    # Collecting the arguments from the argument parser
    args = parser.parse_args()
    # Storing the datasets. Using 3 simulations, with 0,20,40 ENC for single px
    data_0ENC = Xraydata(XRAYRECO_DATA / 'hxsim0ENC_100evts.h5')
   # data_20ENC = Xraydata(XRAYRECO_DATA / 'hxsim_20ENC.h5')
    #data_40ENC = Xraydata(XRAYRECO_DATA / 'hxsim_40ENC.h5')
    # Creating a list in order to loop over datasets
    #datasets = [data_0ENC, data_20ENC, data_40ENC]
    full_input_dataset = []
    full_output_dataset = []
    # Storing all input and output data inside a single overall dataset
    q = multiprocess.Queue()
    q2 = multiprocess.Queue()
    p = multiprocess.Process(target=process_data_worker, args=(data_0ENC, q, q2,))
    p.start()
    p.join()

    while not q.empty():
        full_input_dataset.append(q.get())
    
    print(full_input_dataset)
    '''
    #Dividing the energy and hit position target data (just for clarity)
    target_energies = full_output_dataset[:,0]
    target_xy = full_output_dataset[:,1:]
    
    # Defining the two NNs, one for the energy, one for the position.
    model_e = DNN_e(path_to_weights=args.enweights)
    model_xy = DNN_xy(path_to_weights=args.xyweights)

    # Defining the checkpoint paths
    energy_checkpoint_path = XRAYRECO_ROOT / args.encheckpointpath
    xy_checkpoint_path = XRAYRECO_ROOT / args.xycheckpointpath
    # If the weights files already exist, they are loaded before training 
    # in order to restart training from the last checkpoint
    if energy_checkpoint_path.exists() is True:
        model_e.load_weights(energy_checkpoint_path)
    if xy_checkpoint_path.exists() is True:
        model_xy.load_weights(xy_checkpoint_path)
    
    # Creating the new checkpoints
    cp_callback_e = ModelCheckpoint(filepath=energy_checkpoint_path,
                                    save_weights_only=True, verbose=1)
    cp_callback_xy = ModelCheckpoint(filepath=xy_checkpoint_path,
                                     save_weights_only=True, verbose=1)

    # Training the NNs
    history_e  = model_e.fit(full_input_dataset, target_energies,
                             validation_split=0.05, epochs=50, callbacks=[cp_callback_e])
    history_xy  = model_e.fit(full_input_dataset, target_xy,
                             validation_split=0.05, epochs=50, callbacks=[cp_callback_xy])

    # Plotting the loss trend over epochs for the energy
    plt.figure('Loss over epochs for energy NN')
    plt.plot(history_e.history['val_loss'], label='Validation loss')
    plt.plot(history_e.history['loss'], label='loss')
    plt.legend()

    # Plotting the loss trend over epochs for the hit position
    plt.figure('Loss over epochs for hit position NN')
    plt.plot(history_e.history['val_loss'], label='Validation loss')
    plt.plot(history_e.history['loss'], label='loss')
    plt.legend()
    
    #That's all for this script, the comparison with the std reco is done in another script
    
    #Showing the pictures
    plt.show()
'''
    

