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
import brainModels.datasets as db
from brainModels.datasets import BrainInvaders2015a, ERPCOREN400, erpCoreP300
from brainModels.datasets.base import BaseDataset
from brainModels.datasets.utils import dataset_list
from brainModels.preprocessing import ERP

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
    for key, value in mne.get_config().items():
        print(f"{key}: {value}")
    unittest.main()

    

