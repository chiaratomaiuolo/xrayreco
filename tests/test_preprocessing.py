import unittest

from hexsample.hexagon import HexagonalGrid, HexagonalLayout
from xrayreco.dataprocessing import adc_channel_odd_r, circular_crown_logical_coordinates, \
                            processing_training_data, Xraydata

SAMPLE_PIXELS = ((6, 6), (10, 2), (4, 2), (2, 6), (13, 2), (7, 2), (1, 2))
ORDERINGS_DICT = {0: [0, 5, 1, 3, 2, 6, 4],
                  1: [1, 6, 2, 4, 3, 0, 5],
                  2: [2, 0, 3, 5, 4, 1, 6],
                  3: [3, 1, 4, 6, 5, 2, 0],
                  4: [4, 2, 5, 0, 6, 3, 1],
                  5: [5, 3, 6, 1, 0, 4, 2],
                  6: [6, 4, 0, 2, 1, 5, 3]}

class TestTrackReordering(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        # Initializing a 10x15 ODD_R hexagonal grid for testing
        super(TestTrackReordering, self).__init__(*args, **kwargs)
        self.grid = HexagonalGrid(layout=HexagonalLayout.ODD_R, num_cols=10,\
                             num_rows=15, pitch=60.)
    def test_adc_reordering(self):
        for i in range(7):
            # Storing of pixel's logical coordinates...
            c, r = SAMPLE_PIXELS[i]
            coordinates = circular_crown_logical_coordinates(c, r, self.grid)
            # ... conversion from logical to ADC coordinates for standardizing the order ...
            adc_channel_order = [adc_channel_odd_r(_col, _row) for _col, _row in coordinates]
            self.assertEqual(ORDERINGS_DICT[i], adc_channel_order)
            print(f'Ordering for track having central pixel {i} checked!')

class TestPreprocessingChain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """ Class constructor.
        The constructor opens a specific simulation and extrapolates the signal
        data and the MC truth using the specific methods of the Xraydata class.
        """
        super(TestPreprocessingChain, self).__init__(*args, **kwargs)
        file_path = '/Users/chiara/hexsampledata/sim_HexagonalLayout.ODD_Rum_20enc_srcsigma200um.h5'
        self.data = Xraydata(file_path)
        self.input_data, target_data = processing_training_data(self.data)
        self.target_data_energies = target_data[:,0]
        self.target_data_xy = target_data[:,1:]
    
    def test_data_lenght(self):
        """Test for asserting the lenght of every preprocessed event of a simulation.
        This test checks if every preprocessed input event array of the simulation
        has len == 7 as expected by a circular track.
        """
        # Preprocessing a .h5 simulation file and extrapolating the input data 
        # for the neural network
        for evt in self.input_data:
            self.assertEqual(len(evt), 7)
        self.data.close_file()
    
    def test_targetevts_len(self):
        """Test for asserting the lenght of every preprocessed event of a simulation.
        This test checks if every preprocessed target event array has len == 3,
        corresponding to [energy, x_hit, y_hit] of the incident x-ray.
        """
        # Preprocessing a .h5 simulation file and extrapolating the input data 
        # for the neural network
        for evt_e, evt_xy in zip(self.target_data_energies, self.target_data_xy):
            # Asserting that energies are scalars
            self.assertTrue(not hasattr(evt_e, '__len__'))
            self.assertEqual(len(evt_xy), 2)
        self.data.close_file()
    
    def test_printing_events(self):
        """Test for printing few preprocessed events in order to check if the 
        content is as expected.
        """
        # Printing the first 3 events of the file for checking whether everything
        # is as expected.
        for n in range(3):
            print(f'Event number {n}: input data = {self.input_data[n]}')
            print(f'Event number {n}: target data = {self.target_data_energies[n]},\
                   {self.target_data_xy[n]}')
        self.data.close_file()

    

if __name__ == "__main__":
    # Running the unittests
    unittest.main()
    