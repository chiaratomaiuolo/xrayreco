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