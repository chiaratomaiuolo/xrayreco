
import numpy as np
import tables
from hexsample.fileio import DigiInputFileCircular
from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from hexsample.readout import HexagonalReadoutMode

"""Preprocessing functions.
"""

class Xraydata():
    """Class containing the raw and rearranged data for an X-ray event.
    """
    def __init__(self, input_file_path: str):
        """Constructor
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
        self.grid = HexagonalGrid(HexagonalLayout(layout), numcolumns, numrows, pitch)
        # Extrapolation of data from the DigiEventCircular events
        self.pha = []
        self.columns =[]
        self.rows = []
        for event in input_file:
            self.pha.append(event.pha)
            self.columns.append(event.column)
            self.rows.append(event.row)
        
        # Closing of the input file
        input_file.close()

    def events_formatting(self) -> np.ndarray:
        """This function returns a numpy array containing in every row the events
        data in the following order:
        """
    

        





if __name__ == "__main__":
    input_file_path = '/Users/chiara/hexsampledata/sim_HexagonalLayout.ODD_Rum_20enc_srcsigma200um.h5'
    data = Xraydata(input_file_path)
    pha = data.pha
    columns = data.columns
    rows = data.rows
    print(pha[1003])
    print(columns[1003])
    print(rows[1003])
