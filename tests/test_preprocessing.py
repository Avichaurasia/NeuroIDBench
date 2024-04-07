import inspect
import logging
import shutil
import tempfile
import unittest
import mne
import numpy as np
import pytest
import sys
from mne import BaseEpochs
from mne.io import BaseRaw
sys.path.append("../")
import neuroIDBench.datasets as db
from neuroIDBench.datasets import BrainInvaders2015a, ERPCOREN400, erpCoreP300, dummyDataset
from neuroIDBench.datasets.base import BaseDataset
from neuroIDBench.datasets.utils import dataset_list
from neuroIDBench.preprocessing import BaseERP, ERP

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

class SimpleERP(BaseERP):  # Needed to assess BaseP300
    def used_events(self, dataset):
        return dataset.event_id

class Test_ERP(unittest.TestCase):
    def test_match_all(self):
        # Note: the match all property is implemented in the base paradigm.
        # Thus, although it is located in the P300 section, this test stands for all paradigms.
        paradigm = SimpleERP()
        dataset1 = dummyDataset(
            paradigm="erp",
            event_list=["Target", "NonTarget"],
            channels=["C3", "Cz", "Fz"],
            sfreq=64,
        )
        dataset2 = dummyDataset(
            paradigm="erp",
            event_list=["Target", "NonTarget"],
            channels=["C3", "C4", "Cz"],
            sfreq=256,
        )
        dataset3 = dummyDataset(
            paradigm="erp",
            event_list=["Target", "NonTarget"],
            channels=["C3", "Cz", "Fz", "C4"],
            sfreq=512,
        )
        shift = -0.5

        paradigm.match_all(
            [dataset1, dataset2, dataset3], shift=shift, channel_merge_strategy="union"
        )
        # match_all should returns the smallest frequency minus 0.5.
        # See comment inside the match_all method
        self.assertEqual(paradigm.resample, 64 + shift)
        self.assertEqual(paradigm.channels.sort(), ["C3", "Cz", "Fz", "C4"].sort())
        self.assertEqual(paradigm.interpolate_missing_channels, True)
        X, _, _ = paradigm.get_data(dataset1, subjects=[1])
        n_channels, _ = X[0].shape
        self.assertEqual(n_channels, 4)

        paradigm.match_all(
            [dataset1, dataset2, dataset3],
            shift=shift,
            channel_merge_strategy="intersect",
        )
        self.assertEqual(paradigm.resample, 64 + shift)
        self.assertEqual(paradigm.channels.sort(), ["C3", "Cz"].sort())
        self.assertEqual(paradigm.interpolate_missing_channels, False)
        X, _, _ = paradigm.get_data(dataset1, subjects=[1])
        n_channels, _ = X[0].shape
        self.assertEqual(n_channels, 2)

    def test_BaseERP_paradigm(self):
        paradigm = SimpleERP()
        dataset = dummyDataset(paradigm="erp", event_list=["Deviant", "Standard"])
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # we should have all the same length
        self.assertEqual(len(X), len(labels), len(metadata))
        # X must be a 3D Array
        self.assertEqual(len(X.shape), 3)
        # labels must contain 2 values (Target/NonTarget)
        self.assertEqual(len(np.unique(labels)), 2)
        # metadata must have subjets, sessions, runs
        self.assertTrue("subject" in metadata.columns)
        self.assertTrue("session" in metadata.columns)
        self.assertTrue("run" in metadata.columns)
        # we should have only one subject in the metadata
        self.assertEqual(np.unique(metadata.subject), 1)
        # we should have two sessions in the metadata
        self.assertEqual(len(np.unique(metadata.session)), 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)
        # should return raws
        raws, _, _ = paradigm.get_data(dataset, subjects=[1], return_raws=True)
        for raw in raws:
            self.assertIsInstance(raw, BaseRaw)
        # should raise error
        self.assertRaises(
            ValueError,
            paradigm.get_data,
            dataset,
            subjects=[1],
            return_epochs=True,
            return_raws=True,
        )

    def test_BaseERP_channel_order(self):
        """Test if paradigm return correct channel order, see issue #227."""
        datasetA = dummyDataset(
            paradigm="erp",
            channels=["C3", "Cz", "C4"],
            event_list=["Deviant", "Standard"],
        )
        datasetB = dummyDataset(
            paradigm="p300",
            channels=["Cz", "C4", "C3"],
            event_list=["Deviant", "Standard"],
        )
        paradigm = SimpleERP(channels=["C4", "C3", "Cz"])

        ep1, _, _ = paradigm.get_data(datasetA, subjects=[1], return_epochs=True)
        ep2, _, _ = paradigm.get_data(datasetB, subjects=[1], return_epochs=True)
        self.assertEqual(ep1.info["ch_names"], ep2.info["ch_names"])

    def test_BaseP300_tmintmax(self):
        self.assertRaises(ValueError, SimpleERP, tmin=1, tmax=0)

    def test_BaseP300_filters(self):
        # can work with filter bank
        paradigm = SimpleERP(filters=[[1, 12], [12, 24]])
        dataset = dummyDataset(paradigm="erp", event_list=["Deviant", "Standard"])
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])

        # X must be a 4D Array
        self.assertEqual(len(X.shape), 4)
        self.assertEqual(X.shape[-1], 2)
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

    def test_BaseP300_wrongevent(self):
        # test process_raw return empty list if raw does not contain any
        # selected event. certain runs in dataset are event specific.
        paradigm = SimpleERP(filters=[[1, 12], [12, 24]])
        dataset = dummyDataset(paradigm="p300", event_list=["Deviant", "Standard"])
        epochs_pipeline = paradigm._get_epochs_pipeline(
            return_epochs=True, return_raws=False, dataset=dataset
        )
        # no stim channel after loading cache
        raw = dataset.get_data([1], cache_config=dict(use=False, save_raw=False))[1]["0"][
            "0"
        ]
        raw.load_data()
        self.assertEqual("stim", raw.ch_names[-1])
        # add something on the event channel
        raw._data[-1] *= 10
        with self.assertRaises(ValueError, msg="No events found"):
            epochs_pipeline.transform(raw)
        # zeros it out
        raw._data[-1] *= 0
        with self.assertRaises(ValueError, msg="No events found"):
            epochs_pipeline.transform(raw)

    def test_BaseERP_droppedevent(self):
        dataset = dummyDataset(paradigm="erp", event_list=["Deviant", "Standard"])
        tmax = dataset.interval[1]
        # with regular windows, all epochs should be valid:
        paradigm1 = SimpleERP(tmax=tmax)
        # with large windows, some epochs will have to be dropped:
        paradigm2 = SimpleERP(tmax=10 * tmax)
        # with epochs:
        epochs1, labels1, metadata1 = paradigm1.get_data(dataset, return_epochs=True)
        epochs2, labels2, metadata2 = paradigm2.get_data(dataset, return_epochs=True)
        self.assertEqual(len(epochs1), len(labels1), len(metadata1))
        self.assertEqual(len(epochs2), len(labels2), len(metadata2))
        self.assertGreater(len(epochs1), len(epochs2))
        # with np.array:
        X1, labels1, metadata1 = paradigm1.get_data(dataset)
        X2, labels2, metadata2 = paradigm2.get_data(dataset)
        self.assertEqual(len(X1), len(labels1), len(metadata1))
        self.assertEqual(len(X2), len(labels2), len(metadata2))
        self.assertGreater(len(X1), len(X2))

    def test_BaseERP_epochsmetadata(self):
        dataset = dummyDataset(paradigm="erp", event_list=["Deviant", "Standard"])
        paradigm = SimpleERP()
        epochs, _, metadata = paradigm.get_data(dataset, return_epochs=True)
        # does not work with multiple filters:
        self.assertTrue(metadata.equals(epochs.metadata))

    def test_ERP_specifyevent(self):
        # we can't pass event to this class
        self.assertRaises(ValueError, ERP, events=["a"])

    def test_ERP_wrongevent(self):
        # does not accept dataset with bad event
        paradigm = ERP()
        dataset = dummyDataset(paradigm="erp")
        self.assertRaises(AssertionError, paradigm.get_data, dataset)

    def test_ERP_paradigm(self):
        # with a good dataset
        paradigm = ERP()
        dataset = dummyDataset(event_list=["Deviant", "Standard"], paradigm="erp")
        X, labels, metadata = paradigm.get_data(dataset, subjects=[1])
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertEqual(list(np.unique(labels)), sorted(["Deviant", "Standard"]))
        # should return epochs
        epochs, _, _ = paradigm.get_data(dataset, subjects=[1], return_epochs=True)
        self.assertIsInstance(epochs, BaseEpochs)

if __name__ == "__main__":
    for key, value in mne.get_config().items():
        print(f"{key}: {value}")
    unittest.main()

    

