"""Preprocessing functions.
"""

from typing import Tuple

import numpy as np
from tqdm import tqdm

from hexsample.fileio import DigiInputFileCircular
from hexsample.hexagon import HexagonalGrid, HexagonalLayout


def circular_crown_logical_coordinates(column: int, row: int, grid: HexagonalGrid)\
    -> Tuple[np.ndarray, np.ndarray]:
    """This function returns a set of 7 tuples (col, row) containing the logical
    coordinates of the pixels in a DigiCircularEvent in a standard ordering.
    Consider the following legend: 
    c = central, ur = up-right, r = right, dr = down-right, d = down, 
    dl = down-left, l = left, ul = upper-left.
    The output array has the tuples in the following order:
    [c, ur, r, dr, d, dl, l, ul]

    Arguments
    ---------
    column : int
        central pixel column in pixel logical coordinates;
    row : int
        central pixel row in pixel logical coordinates;
    grid : HexagonalGrid
        class containing the informations about the hexagonal grid features and
        the methods for localizing the neighbors of a given pixel.
    """
    coordinates = [(column, row)] + [(c, r) for c, r in grid.neighbors(column, row)]
    return coordinates

def highest_pixel_coordinates(self) -> np.array:
        """This function returns a numpy array containing the physical coordinates
        of the highest pixel (the one that defines the position of the cluster). 
        This function is not useful for the NN intself, instead for performance 
        evaluation tasks.
        """
        # Creating lists for storing the x and y coordinates
        x = []
        y = []
        for evt in self.input_file:
            x_tmp, y_tmp = self.grid.pixel_to_world(evt.column, evt.row)
            x.append(x_tmp)
            y.append(y_tmp)
        return x, y

class Xraydata():
    """Class that preprocesses data from a .h5 file, creates the input arrays
    for the NN and provides an easy access to simulation information.
    """
    def __init__(self, file_path: str):
        """Class constructor

        Arguments
        ---------
        file_path : str
            String containing the path to the .h5 file to be processed.
        """
        # Initializing the DigiEventCircular from the .h5 input file
        self.input_file = DigiInputFileCircular(file_path)
        # Initializing the HexagonalGrid
        # Extrapolation of informations of the HexagonalGrid from the file header
        layout = self.input_file.root.header._v_attrs['layout']
        numcolumns = self.input_file.root.header._v_attrs['numcolumns']
        numrows = self.input_file.root.header._v_attrs['numrows']
        pitch = self.input_file.root.header._v_attrs['pitch']
        # Construction of the HexagonalGrid. This is needed because the coordinates
        # of the adjacent pixels of a given one depends on the layout.
        self.grid = HexagonalGrid(HexagonalLayout(layout), numcolumns, numrows, pitch)

        # Saving all the necessary columns for input data
        pha = []
        columns = []
        rows = []
        for i, evt in tqdm(enumerate(self.input_file)):
            pha.append(evt.pha)
            rows.append(evt.row)
            columns.append(evt.column)
        # Saving raw data as class members
        self.pha = np.array(pha)
        self.columns = np.array(columns)
        self.rows = np.array(rows)
        # Saving the MC truth arrays as class members
        self.mc_energy = np.array(self.input_file.mc_column('energy'))
        self.mc_x = np.array(self.input_file.mc_column('absx'))
        self.mc_y = np.array(self.input_file.mc_column('absy'))
    
    def __del__(self):
        """Instance destructor. This is needed for the proper closing of the 
        input data file.
        """
        # Closing the input file ...
        self.input_file.close()
        # ... and then deleting the class instance.
    
    def __str__(self):
        """Implementing print(). It prints out the data file name.
        """
        return self.input_file.root.header._v_attrs['outfile']
    
    def __repr__(self) -> str:
        """Implementing repr() method. It shows general information about the
        simulation in file_path stored in the Xraydata object.
        """
        print('General simulation information:')
        group_path = '/header'
        group = self.input_file.get_node(group_path)
        for attr_name in group._v_attrs._f_list():
            print(f'{attr_name}: {group._v_attrs[attr_name]}')
        return ''
    
def processing_data(data: Xraydata) -> Tuple[np.array, np.array]:
    """This function takes as input an Xraydata object and processes its raw data
    in order to extract the input and target datasets to be given to the NN
    for training, evaluation and data prediction (in this case, cleary only the
    input data are given)

    Arguments
    ---------
    data : Xraydata
        Xraydata object containing the raw data
    
    Return
    ------
    input_processed_data : np.array
        Pre-processed input data for the NN. The shape of the array, is (n,7,3),
        where n depends on the number of events in the Xraydata object.
    target_processed_data : np.array
        Pre-processed target data from the MC truth table to be used as target data
        for NN training an evaluation. The shape of the array is (n,3), where 
        n depends on the number of events in the Xraydata object.
    """
    # Defining the input data list to be filled in a for loop
    input_processed_data = []
    # Saving the highest pixel coordinates for target coordinates rescaling.
    x_max = []
    y_max = []
    # Looping on events: extrapolating data and converting into required format
    print(f'Processing raw data events from {data} dataset...\n')
    for p, r, c in tqdm(zip(data.pha, data.columns, data.rows)):
        # Storing of pixel's logical coordinates...
        coordinates = circular_crown_logical_coordinates(c, r, data.grid)
        # ... conversion from logical to ADC coordinates for standardizing the order ...
        adc_channel_order = [data.grid.adc_channel(_col, _row) for _col, _row in coordinates]
        # ... storing the re-ordered PHA list ...
        ordered_p = p[adc_channel_order]
        # ... separating x and y from coordinates tuples ...
        x_logical, y_logical = zip(*coordinates)
        # ... and converting from logical to physical coordinates ...
        # (note that the function pixel_to_world() needs the conversion
        # from tuples to numpy array)
        x, y = data.grid.pixel_to_world(np.array(x_logical), np.array(y_logical))
        # ... then stack the coordinates with its corresponding signal value
        # and append the result to the data list.
        input_processed_data.append(np.stack((list(zip(ordered_p, x-x[0], y-y[0]))), axis=0))

        #Saving the coordinates of the highest pixel for rescaling target data
        x_max.append(x[0])
        y_max.append(y[0])
    # Constructing target data: rescaling positions with respect to the central signal px
    # and zipping energy with (x, y) coordinates.
    processed_target_coordinates = np.stack((list(zip(data.mc_x-x_max, data.mc_y-y_max))), axis=0)
    processed_target_data = np.stack((list(zip(data.mc_energy, data.mc_x-x_max, data.mc_y-y_max))), axis=0)

    # Return the events_data list of arrays.
    return np.array(input_processed_data), processed_target_data

if __name__ == "__main__":
    # Loading an hexsample simulation and storing its content into an Xraydata object
    # The following simulation is a simulation with no electronic noise.
    file_path = '/Users/chiara/hexsampledata/hxsim_20ENC.h5'
    # Creating an istance of the class Xraydata that contains the data preprocessing methods
    data = Xraydata(file_path)
    #Printing the general information about the simulation stored in file_path
    repr(data)
    input_data, target_data = processing_data(data)
    print(input_data[0])
    print(target_data[0])
    del data

