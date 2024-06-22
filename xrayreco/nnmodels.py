''' NN models definition
'''
from typing import Tuple

import numpy as np

from keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from matplotlib import pyplot as plt


def DNN_e(input_shape: Tuple=(7,3), path_to_weights: str=None) -> Model:
    '''Definition of the Deep Neural Network model for energy reconstruction.
    This function defines the chosen architecture for the NN and returns the
    (untrained) keras.Model.

    Arguments
    ---------
    input_shape : Tuple
        Shape of the input in the for of a Tuple, that is the format taken as
        input by keras.Input(). The default value is set to (7,3), that is the
        shape of a photon track of 7 pixels, having for each pixel [pha, x, y].
    path_to_weights : str
        Path to a .weights.h5 file to be loaded on the model (after having checked
        its compatibility with the NN architecture).
    '''
    model_e = Sequential([
                    Flatten(input_shape=(7,3)),
                    BatchNormalization(),
                    Dense(20, activation='relu'),
                    Dense(50, activation='relu'),
                    Dense(50, activation='relu'),
                    Dense(50, activation='relu'),
                    Dense(10, activation='relu'),
                    Dense(1, activation='linear')
                    ])
    # If provided, a .weights.h5 file is loaded into the model
    if path_to_weights is not None:
        #Check compatibility
        try:
            model_e.load_weights(path_to_weights)
        except(ValueError):
            #If the file is not compatible with the shape of the model, 
            # the model without any training is provided (as if 
            # path_to_weights wasn't provided).
            model_e.compile(optimizer='adam',loss='MSE',metrics=['accuracy'])
            return model_e
        model_e.compile(optimizer='adam',loss='MSE',metrics=['accuracy'])
    else:
        model_e.compile(optimizer='adam',loss='MSE',metrics=['accuracy'])
    return model_e


def DNN_xy(input_shape: Tuple=(7,3), path_to_weights: str=None) -> Model:
    '''Definition of the Deep Neural Network model for hit coordinates [x,y] 
    reconstruction. This function defines the chosen architecture for the NN
    and returns the (untrained) keras.Model.
    
    Arguments
    ---------
    input_shape : Tuple
        Shape of the input in the for of a Tuple, that is the format taken as
        input by keras.Input(). The default value is set to (7,3), that is the
        shape of a photon track of 7 pixels, having for each pixel [pha, x, y].
    path_to_weights : str
        Path to a .weights.h5 file to be loaded on the model (after having checked
        its compatibility with the NN architecture).
    '''
    model_xy = Sequential([
                    Input(shape=(7,3)),
                    Flatten(),
                    BatchNormalization(),
                    Dense(30, activation='relu'),
                    Dense(50, activation='relu'),
                    Dense(100, activation='relu'),
                    Dense(50, activation='relu'),
                    Dense(30, activation='relu'),
                    Dense(2, activation='linear')
                    ])
    # If provided, a .weights.h5 file is loaded into the model
    if path_to_weights is not None:
        #Check compatibility
        try:
            model_xy.load_weights(path_to_weights)
        except(ValueError):
            #If the file is not compatible with the shape of the model, 
            # the model without any training is provided (as if 
            # path_to_weights wasn't provided).
            model_xy.compile(optimizer='adam',loss='MSE',metrics=['accuracy'])
            return model_xy
        model_xy.compile(optimizer='adam',loss='MSE',metrics=['accuracy'])
    else:
        model_xy.compile(optimizer='adam',loss='MSE',metrics=['accuracy'])
    return model_xy


def e_training_worker(input_data: np.array, target_data: np.array,
                      checkpoint_path: str = None, **kwargs):
    """Defining a training worker for multiprocess purposes
    """
    model_e = DNN_xy(path_to_weights=checkpoint_path)
    # If the weights files already exist, they are loaded before training 
    # in order to restart training from the last checkpoint
    if checkpoint_path.exists() is True:
        model_e.load_weights(checkpoint_path)
    
    # Creating the new checkpoints
    cp_callback_e = ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True, verbose=1)
    
    # Training the NN 
    history_e = model_e.fit(input_data, target_data, validation_split=0.05, epochs=20, callbacks=[cp_callback_e])
    return history_e

def xy_training_worker(input_data: np.array, target_data: np.array,
                      checkpoint_path: str = None, **kwargs):
    """Defining a training worker for multiprocess purposes
    """
    model_xy = DNN_e(path_to_weights=checkpoint_path)
    # If the weights files already exist, they are loaded before training 
    # in order to restart training from the last checkpoint
    if checkpoint_path.exists() is True:
        model_xy.load_weights(checkpoint_path)
    
    # Creating the new checkpoints
    cp_callback_xy = ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True, verbose=1)
    
    # Training the NN 
    history_xy = model_xy.fit(input_data, target_data, validation_split=0.05, epochs=20, callbacks=[cp_callback_xy])
    return history_xy









