
import numpy as np
from tqdm import tqdm
from typing import Tuple

from hexsample.fileio import DigiInputFileCircular
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.readout import HexagonalReadoutMode

"""Preprocessing functions.
"""

def circular_crown_logical_coordinates(column: int, row: int, grid: HexagonalGrid) -> Tuple[np.ndarray, np.ndarray]:
    """This function returns a set of 7 tuples (col, row) containing the logical
    coordinates of the pixels in a DigiCircularEvent in a standard ordering.
    Consider the following legend: 
    c = central, ur = up-right, r = right, dr = down-right, d = down, 
    dl = down-left, l = left, ul = upper-left.
    The output array has the tuples in the following order:
    [c, ur, r, dr, d, dl, l, ul]
    """
    central_coordinates = [(column, row)]
    coordinates = central_coordinates + [(c, r) for c, r in grid.neighbors(column, row)]
    return coordinates


def input_events_data(input_file_path: str) -> list:
    """This function returns a numpy array containing in every row the data 
    of a single event to be given as input to the neural network.
    Before stacking data inside an array, it is necessary to preprocess them.
    Central pixel position is converted from logical coordinates (col, row) to 
    cartesian coordinates (x_max,y_max), PHA array is rearranged in a standard ordering. 
    Consider the following legend: 
    ur = up-right, r = right, dr = down-right, d = down, dl = down-left, l = left, ul = upper-left.
    The output array has the data in the following order:
    [central pha, ur pha, r pha, dr pha, d pha, dl pha, l pha, ul pha, central x, central y]
    """
    # Extrapolation of the DigiEventCircular from the .h5 input file
    input_file = DigiInputFileCircular(input_file_path)
    # Extrapolation of informations of the HexagonalGrid from the file header
    layout = input_file.root.header._v_attrs['layout']
    numcolumns = input_file.root.header._v_attrs['numcolumns']
    numrows = input_file.root.header._v_attrs['numrows']
    pitch = input_file.root.header._v_attrs['pitch']
    # Construction of the HexagonalGrid. This is needed because the coordinates
    # of the adjacent pixels of a given one depends on the layout.
    grid = HexagonalGrid(HexagonalLayout(layout), numcolumns, numrows, pitch)
    # Extrapolation of data from the DigiEventCircular events
    events_data = []
    # Looping on events: extrapolating data and converting into required format
    for i, event in tqdm(enumerate(input_file)):
        # Pixel logical coordinates storing...
        coordinates = circular_crown_logical_coordinates(event.column, event.row, grid)
        # ... conversion from logical to ADC coordinates to standardize the order ...
        adc_channel_order = [grid.adc_channel(_col, _row) for _col, _row in coordinates]
        # ... storing the re-ordered PHA list ...
        pha = event.pha[adc_channel_order]
        # ... separating x and y from coordinates tuples ...
        x_logical, y_logical = zip(*coordinates)
        # ... and converting from logical to physical coordinates ...
        # (note that the function pixel_to_world() needs the conversion
        # from tuples to numpy array)
        x, y = grid.pixel_to_world(np.array(x_logical),np.array(y_logical))
        # ... then stack the coordinates with its corresponding signal value
        # and append the result to the data list.
        events_data.append(np.stack((zip(x, y, pha)), axis=0))
    # Closing the input file
    input_file.close()
    # Return the events_data list of arrays.
    return np.array(events_data)


if __name__ == "__main__":
    input_file_path = '/Users/chiara/hexsampledata/sim_HexagonalLayout.ODD_Rum_20enc_srcsigma200um.h5'
    data = input_events_data(input_file_path)
    print(data[1])
