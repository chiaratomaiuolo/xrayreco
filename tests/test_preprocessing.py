import unittest

from xrayreco.preprocessing import Xraydata


class TestPreprocessingChain(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """ Class constructor.
        The constructor opens a specific simulation and extrapolates the signal
        data and the MC truth using the specific methods of the Xraydata class.
        """
        super(TestPreprocessingChain, self).__init__(*args, **kwargs)
        file_path = '/Users/chiara/hexsampledata/sim_HexagonalLayout.ODD_Rum_20enc_srcsigma200um.h5'
        self.data = Xraydata(file_path)
        self.input_data = self.data.input_events_data()
        self.target_data_energies, self.target_data_xy = self.data.target_data()
    
    def test_data_lenght(self):
        """Test for asserting the lenght of every preprocessed event of a simulation.
        This test checks if every preprocessed input event array of the simulation
        has len == 7 as expected by a circular track.
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
        for evt_e, evt_xy in zip(self.target_data_energies, self.target_data_xy):
            self.assertTrue(not hasattr(evt_e, '__len__'))
            self.assertEqual(len(evt_xy), 2)
    
    def test_printing_events(self):
        """Test for printing few preprocessed events in order to check if the 
        content is as expected.
        """
        # Printing the first 3 events of the file for checking whether everything
        # is as expected.
        for n in range(3):
            print(f'Event number {n}: input data = {self.input_data[n]}')
            print(f'Event number {n}: target data = {self.target_data_energies[n]}, {self.target_data_xy[n]}')
    

if __name__ == "__main__":
    # Running the unittests
    unittest.main()
    