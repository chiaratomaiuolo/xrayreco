import unittest

from Xray_hit_reconstruction.preprocessing import Xraydata


class TestPreprocessingChain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """ Class constructor.
        """
        print('constructor')
        super(TestPreprocessingChain, self).__init__(*args, **kwargs)
        file_path = '/Users/chiara/hexsampledata/sim_HexagonalLayout.ODD_Rum_20enc_srcsigma200um.h5'
        self.data = Xraydata(file_path)
        self.input_data = self.data.input_events_data()
        self.target_data = self.data.target_data()

    def test_inputevts_len(self):
        """Test for asserting the lenght of every preprocessed event of a simulation.
        This test checks if every preprocessed input event array of the simulation has len == 7 
        as expected by a circular track size.
        """
        # Preprocessing a .h5 simulation file and extrapolating the input data 
        # for the neural network
        for evt in self.input_data:
            self.assertEqual(len(evt), 7)
    
    def test_targetevts_len(self):
        """Test for asserting the lenght of every preprocessed event of a simulation.
        This test checks if every preprocessed target event array has len == 3,
        corresponding to [energy, x_hit, y_hit] of the incident x-ray.
        """
        # Preprocessing a .h5 simulation file and extrapolating the input data 
        # for the neural network
        for evt in self.target_data:
            self.assertEqual(len(evt), 3)
    
    def printing_events(self):
        """Test for printing few preprocessed events in order to check if the 
        content is as expected.
        """
        # Printing the first 3 events of the file for checking whether everything
        # is as expected.
        for n in range(3):
            print(f'Event number {n}: input data = {self.input_data[n]}')
            print(f'Event number {n}: target data = {self.target_data[n]}')


def test_preprocessing(file_path: str):
    """ Test for the preprocessing of simulated events data.

    Arguments
    ---------
    - file_path : str
        file path to the .h5 file to be transformed into NN training data.
    """
    # Transforming the .h5 file into an Xraydata object
    data = Xraydata(file_path)
    input_data = data.input_events_data()
    output_data = data.target_data()
    # Printing the first 3 events of the file for checking whether everything
    # is as expected.
    for n in range(3):
        print(f'Event number {n}: input data = {input_data[n]}, len = {len(input_data[n])}')
        print(f'Event number {n}: target data = {output_data[n]}, len = {len(output_data[n])}')
    # Deleting the data object
    print('Deleting data object after the test')
    del data

if __name__ == "__main__":
    # Running the unittests
    unittest.main()
    