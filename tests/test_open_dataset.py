import inspect
import logging
import shutil
import tempfile
import unittest
import mne
import numpy as np
import pytest
import sys
sys.path.append("../")
import neuroIDBench.datasets as db
from neuroIDBench.datasets import BrainInvaders2015a, ERPCOREN400, erpCoreP300
from neuroIDBench.datasets.base import BaseDataset
from neuroIDBench.datasets.utils import dataset_list
from neuroIDBench.preprocessing import ERP

_ = mne.set_log_level("CRITICAL")

def _run_tests_on_dataset(d):
    """Run tests on a dataset object."""

    for sub in d.suject_list:

        # check that we can get data
        data = d.get_data(subjects=[sub])

        # we should get a dict
        assert isinstance(data, dict)

        # We should get a raw array and events at the end

        rawdata, events = tuple(data[sub]["0"]["0"]).values()
        assert issubclass(type(rawdata), mne.io.BaseRaw), type(rawdata)
        assert issubclass(type(events), np.ndarray), type(events)
        assert events.shape[1] == 3, events.shape

class Test_Datasets(unittest.TestCase):
    def setUp(self):
        self.dataset = BrainInvaders2015a()

    def test_invalid_subject(self):
        with self.assertRaises(ValueError):  # assuming ValueError for invalid subject
            self.dataset.get_data(subjects=['1'])

    # Test invlaid session name, session name should "session_1, session_2, session_3". Basically, it should not be an integer
    def test_invalid_session(self):
        rawdata, events = tuple(self.dataset.get_data(subjects=[1]).values())[0].values()
        with self.assertRaises(ValueError):
            # if rawdata which is dictionary has second key not as session_1, session_2, session_3, then it should raise an error
            rawdata[1]['1']
            


    def test_empty_data(self):
        with self.assertRaises(ValueError):  # assuming ValueError for empty data
            self.dataset.get_data(subjects=0)
    
    def test_dataset_accept(self):
        """Verify that accept licence is working."""
        for ds in [BrainInvaders2015a()]:
            # if the data is already downloaded:
            if mne.get_config("MNE_DATASETS_BRAININVADERS2015A_PATH") is None:
                self.assertRaises(AttributeError, ds.get_data, [1])

    def test_dataset_list(self):
        all_datasets = [
            c
            for c in db.__dict__.values()
            if (
                inspect.isclass(c)
                and issubclass(c, BaseDataset)
            )
        ]
        assert len(dataset_list) == len(all_datasets)
        assert set(dataset_list) == set(all_datasets)

if __name__ == "__main__":
    unittest.main()

    

