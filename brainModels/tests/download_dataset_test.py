"""Tests to ensure that datasets download correctly."""
import unittest
import mne

from moabb.datasets.bbci_eeg_fnirs import BaseShin2017

class Test_Downloads(unittest.TestCase):
    def run_dataset(self, dataset, subj=(0, 2)):
        def _get_events(raw):
            stim_channels = mne.utils._get_stim_channel(None, raw.info, raise_error=False)
            if len(stim_channels) > 0:
                events = mne.find_events(raw, shortest_event=0, verbose=False)
            else:
                events, _ = mne.events_from_annotations(raw, verbose=False)
            return events

        if isinstance(dataset(), BaseShin2017):
            obj = dataset(accept=True)
        else:
            obj = dataset()
        obj.subject_list = obj.subject_list[subj[0] : subj[1]]
        data = obj.get_data(obj.subject_list)

        # get data return a dict
        self.assertTrue(isinstance(data, dict))

        # keys must corresponds to subjects list
        self.assertTrue(list(data.keys()) == obj.subject_list)

        # session must be a dict, and the length must match
        for _, sessions in data.items():
            self.assertTrue(isinstance(sessions, dict))
            self.assertTrue(len(sessions) >= obj.n_sessions)

            # each session is a dict, with multiple runs
            for _, runs in sessions.items():
                self.assertTrue(isinstance(runs, dict))

                for _, raw in runs.items():
                    self.assertTrue(isinstance(raw, mne.io.BaseRaw))

                # each raw should contains events
                for _, raw in runs.items():
                    self.assertTrue(len(_get_events(raw) != 0))

if __name__ == "__main__":
    unittest.main()