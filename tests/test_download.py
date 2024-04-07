"""Tests to ensure that datasets download correctly."""
import unittest
import mne
import numpy as np
import sys
sys.path.append("../")
from neuroIDBench.datasets import (BrainInvaders2015a, ERPCOREN400, COGBCIFLANKER, Mantegna2019)
#from ..datasets import (BrainInvaders2015a, ERPCOREN400, COGBCIFLANKER, Mantegna2019)


class TestDownloads(unittest.TestCase):
    @staticmethod
    def get_events(raw):
        """Get events from raw data.

        Parameters:
        -----------
        raw: mne.io.BaseRaw
            raw data    
        """

        stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
        if stim_channels:
            return mne.find_events(raw, shortest_event=0, verbose=False)
        else:
            return mne.events_from_annotations(raw, verbose=False)
        

    def run_dataset(self, dataset, subjects=(0, 2)):
        """Run tests on a dataset object.

        Parameters:
        -----------
        dataset: brainModels.datasets
            dataset to test
            subjects: tuple (range of subjects to test)       
        """

        print(f"Testing {dataset.__name__} dataset")
        if isinstance(dataset, BrainInvaders2015a):
            data_loader = dataset(accept=True)
        else:
            data_loader = dataset()
        data_loader.subject_list = data_loader.subject_list[subjects[0]:subjects[1]]
        data = data_loader.get_data(data_loader.subject_list)

        #data = _load_data(dataset)

        # get data return a dict
        self.assertIsInstance(data, dict)

        # keys must corresponds to subjects list
        self.assertEqual(list(data.keys()), data_loader.subject_list)

        # session must be a dict, and the length must match
        for subject, sessions in data.items():
            self.assertIsInstance(sessions, dict)

            # each session is a dict, with multiple runs
            self.assertGreaterEqual(len(sessions), data_loader.n_sessions)

            # each session is a dict, with multiple runs
            for session, runs in sessions.items():
                self.assertIsInstance(runs, dict)

                # each raw should contains mne raw data and events
                for run, raw_object in runs.items():
                    raw, events = raw_object
                    self.assertIsInstance(raw, mne.io.BaseRaw)
                    self.assertGreater(len(events), 0)

                    # check that events are sorted
                    assert np.all(np.diff(events[:, 1]) > 0), events[:, 1]

if __name__ == "__main__":
    unittest.main()