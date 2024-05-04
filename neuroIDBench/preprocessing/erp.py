"""ERP Paradigms"""
import abc
import logging
import mne
import numpy as np
import pandas as pd
from .base import BaseParadigm
from ..datasets import utils

log = logging.getLogger(__name__)


class BaseERP(BaseParadigm):
    """Base P300 paradigm.

    Please use one of the child classes

    Parameters
    ----------

    filters: list of list (defaults [[7, 35]])
        bank of bandpass filter to apply.

    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    baseline: None | tuple of length 2
            The time interval to consider as “baseline” when applying baseline
            correction. If None, do not apply baseline correction.
            If a tuple (a, b), the interval is between a and b (in seconds),
            including the endpoints.
            Correction is applied by computing the mean of the baseline period
            and subtracting it from the data (see mne.Epochs)

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.
    """

    def __init__(
        self,
        filters=([1, 50],),
        events=None,
        tmin=0.0,
        tmax=None,
        baseline=(None,0),
        channels=None,
        resample=None,
        reject=False,
    ):
        super().__init__()
        self.filters = filters
        self.events = events
        self.channels = channels
        self.baseline = baseline
        self.resample = resample
        self.reject = reject

        if tmax is not None:
            if tmin >= tmax:
                raise (ValueError("tmax must be greater than tmin"))

        self.tmin = tmin
        self.tmax = tmax

    def is_valid(self, dataset):
        ret = True
        if not (dataset.paradigm == "erp") :
            ret = False

        # check if dataset has required events
        if not dataset.event_id:
            ret=False
            
        # we should verify list of channels, somehow
        return ret

    @abc.abstractmethod
    def used_events(self, dataset):
        pass

    @property
    def datasets(self):
        if self.tmax is None:
            interval = None
        else:
            interval = self.tmax - self.tmin
        return utils.dataset_search(
            paradigm='erp', events=self.events, interval=interval, has_all_events=True
        )

    @property
    def scoring(self):
        return "roc_auc"


class SinglePass(BaseERP):
    """Single Bandpass filter P300

    P300 paradigm with only one bandpass filter (default 1 to 24 Hz)

    Parameters
    ----------
    fmin: float (default 1)
        cutoff frequency (Hz) for the high pass filter

    fmax: float (default 24)
        cutoff frequency (Hz) for the low pass filter

    events: List of str | None (default None)
        event to use for epoching. If None, default to all events defined in
        the dataset.

    tmin: float (default 0.0)
        Start time (in second) of the epoch, relative to the dataset specific
        task interval e.g. tmin = 1 would mean the epoch will start 1 second
        after the beginning of the task as defined by the dataset.

    tmax: float | None, (default None)
        End time (in second) of the epoch, relative to the beginning of the
        dataset specific task interval. tmax = 5 would mean the epoch will end
        5 second after the beginning of the task as defined in the dataset. If
        None, use the dataset value.

    baseline: None | tuple of length 2
            The time interval to consider as “baseline” when applying baseline
            correction. If None, do not apply baseline correction.
            If a tuple (a, b), the interval is between a and b (in seconds),
            including the endpoints.
            Correction is applied by computing the mean of the baseline period
            and subtracting it from the data (see mne.Epochs)

    channels: list of str | None (default None)
        list of channel to select. If None, use all EEG channels available in
        the dataset.

    resample: float | None (default None)
        If not None, resample the eeg data with the sampling rate provided.

    """

    def __init__(self, fmin=1, fmax=50, **kwargs):
        if "filters" in kwargs.keys():
            raise (ValueError("ERP does not take argument filters"))
        super().__init__(filters=[[fmin, fmax]], **kwargs)


class ERP(SinglePass):
    """P300 for Target/NonTarget classification

    Metric is 'roc_auc'

    """
    def __init__(self, **kwargs):
        if "events" in kwargs.keys():
            raise (ValueError("ERP dont accept events"))
        super().__init__(events=["Deviant", "Standard"], **kwargs)

    def used_events(self, dataset):
        return dataset.event_id

    @property
    def scoring(self):
        return "roc_auc"
    
