"""Preprocessing functions.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import tables
from tqdm import tqdm

from hexsample.fileio import DigiInputFileCircular, FileType, OutputFileBase, \
                             ReconInputFile
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
    coordinates = [(column, row)] + list(grid.neighbors(column, row))
    return coordinates

# pylint: disable=locally-disabled, too-many-instance-attributes, unused-variable
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
    
    def close_file(self):
        """Closing the input file if needed
        """
        self.input_file.close()

# pylint: disable=locally-disabled, unused-variable
def processing_training_data(data: Xraydata) -> Tuple[np.array, np.array]:
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
    for p, c, r in tqdm(zip(data.pha, data.columns, data.rows)):
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
    processed_target_data = np.stack((list(zip(data.mc_energy, data.mc_x-x_max,
                                               data.mc_y-y_max))), axis=0)

    # Return the events_data list of arrays.
    return np.array(input_processed_data), processed_target_data

# pylint: disable=locally-disabled, unused-variable
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
    """
    # Defining the input data list to be filled in a for loop
    input_processed_data = []
    # Looping on events: extrapolating data and converting into required format
    print(f'Processing raw data events from {data} dataset...\n')
    for p, c, r in tqdm(zip(data.pha, data.columns, data.rows)):
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

    # Return the events_data list of arrays.
    return np.array(input_processed_data)

def highest_pixel_coordinates(data: Xraydata) -> np.array:
    """This function returns a numpy array containing the physical coordinates
    of the highest pixel (the one that defines the position of the cluster). 
    This function is not useful for the NN intself, instead for performance 
    evaluation tasks.
    """
    # Creating lists for storing the x and y coordinates
    x = []
    y = []
    for evt in data.input_file:
        x_tmp, y_tmp = data.grid.pixel_to_world(evt.column, evt.row)
        x.append(x_tmp)
        y.append(y_tmp)
    return x, y

def recon_data(recon_file_path: str) -> Tuple[np.array, np.array, np.array]:
    """ Extracts the reconstructed quantities from a ReconInputFile.
    Returns 3 arrays that are, respectively:
    [reconstructed energy, reconstructed x_hit, reconstructed y_hit]
    Arguments
    ---------
    recon_file_path : str
        path to ReconInputFile

    """
    recon_file = ReconInputFile(recon_file_path)
    energy, x, y = recon_file.column('energy'),\
                   recon_file.column('posx'),\
                   recon_file.column('posy')
    # Closing file
    recon_file.close()
    return energy, x, y

"""
-------------------- Datafile structures ---------------------
"""

@dataclass
class PredEvent:

    """Descriptor for a reconstructed event.

    Arguments
    ---------
    trigger_id : int
        The trigger identifier.

    timestamp : float
        The timestamp (in s) of the event.

    livetime : int
        The livetime (in us) since the last event.

    roi_size : int
        The ROI size for the event.

    cluster : Cluster
        The reconstructed cluster for the event.
    """

    trigger_id: int
    timestamp: float
    livetime: int
    energy: float
    posx: float
    posy: float

class PredDescription(tables.IsDescription):

    """Description of the pred file format.
    """

    # pylint: disable=too-few-public-methods

    trigger_id = tables.Int32Col(pos=0)
    timestamp = tables.Float64Col(pos=1)
    livetime = tables.Int32Col(pos=2)
    energy = tables.Float32Col(pos=3)
    posx = tables.Float32Col(pos=4)
    posy = tables.Float32Col(pos=5)

def _fill_pred_row(row: tables.tableextension.Row, event: PredEvent) -> None:
    """Helper function to fill an output table row, given a ReconEvent object.
    .. note::
    This would have naturally belonged to the ReconDescription class as
    a @staticmethod, but doing so is apparently breaking something into the
    tables internals, and all of a sudden you get an exception due to the
    fact that a staticmethod cannot be pickled.
    """
    row['trigger_id'] = event.trigger_id
    row['timestamp'] = event.timestamp
    row['livetime'] = event.livetime
    row['energy'] = event.energy
    row['posx'] = event.posx
    row['posy'] = event.posy
    row.append()

class PredictedOutputFile(OutputFileBase):

    """Description of a predicted output file. The structure and methods are taken
    from the ones of ReconEvents of hexsample library.

    Arguments
    ---------
    file_path : str
        The path to the file on disk.
    """

    _FILE_TYPE = FileType.RECON
    PRED_TABLE_SPECS = ('pred_table', PredDescription, 'Predicted data')

    def __init__(self, file_path: str):
        """Constructor.
        """
        super().__init__(file_path)
        self.digi_header_group = self.create_group(self.root, 'digi_header', 'Digi file header')
        self.pred_group = self.create_group(self.root, 'pred', 'Predicted')
        self.pred_table = self.create_table(self.pred_group, *self.PRED_TABLE_SPECS)

    def update_digi_header(self, **kwargs):
        """Update the user arguments in the digi header group.
        """
        self.update_user_attributes(self.digi_header_group, **kwargs)

    def add_row(self, pred_event: PredEvent) -> None:
        """Add one row to the file.

        Arguments
        ---------
        digi : PredEvent
            The predicted event contribution.
        """
        # pylint: disable=arguments-differ
        _fill_pred_row(self.pred_table.row, pred_event)

    def flush(self) -> None:
        """Flush the basic file components.
        """
        self.pred_table.flush()