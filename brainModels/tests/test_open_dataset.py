import inspect
import logging
import shutil
import tempfile
import unittest
import mne
import numpy as np
import pytest
import brainModels.datasets as datasets
from brainModels.datasets import (BrainInvaders2015a, ERPCOREN400, COGBCIFLANKER, Mantegna2019)#
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


class TestDatasets(unittest.TestCase):
    """Tests for all datasets."""

    def test_dataset_accept(self):
        """Verify that accept licence is working."""
        # Only BaseShin2017 (bbci_eeg_fnirs) for now
        for ds in [BrainInvaders2015a(), ERPCOREN400()]:
            # if the data is already downloaded:
            if mne.get_config("MNE_DATASETS_BBCIFNIRS_PATH") is None:
                self.assertRaises(AttributeError, ds.get_data, [1])



        # print events
        #print(mne.find_events(rawdata))
        #print(d.event_id)
    

