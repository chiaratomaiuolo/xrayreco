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
                    Input(shape=input_shape),
                    # Flattening the input
                    Flatten(),
                    # Performing the batch normalization
                    BatchNormalization(),
                    Dense(64, activation='relu'),
                    Dense(256, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(32, activation='relu'),
                    # Returning a single output
                    # Using linear activation func for the regression task
                    # Missing a Dropout layer, ask Rizzi []
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
            return model_e
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
                    Input(shape=input_shape),
                    # Flattening the input
                    Flatten(),
                    # Performing the batch normalization
                    BatchNormalization(),
                    Dense(64, activation='relu'),
                    Dense(256, activation='relu'),
                    Dense(128, activation='relu'),
                    Dense(64, activation='relu'),
                    Dense(32, activation='relu'),
                    # Returning two outputs: (x, y)
                    # Using linear activation func for the regression task
                    # Missing a Dropout layer, ask Rizzi []
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
            return model_xy
    return model_xy









