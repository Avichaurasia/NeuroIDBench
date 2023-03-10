#!/usr/bin/env python
# coding: utf-8
import glob
import os
import os.path as osp
from pathlib import Path
import shutil
import zipfile as z
from distutils.dir_util import copy_tree
import mne
import numpy as np
import yaml
from mne.channels import make_standard_montage
from scipy.io import loadmat
from deeb.datasets import download as dl
from deeb.datasets.base import BaseDataset
from collections import OrderedDict
#import patoolib
#import patoolib
#from pyunpack import Archive
from mne.utils import _url_to_local_path, verbose
import shutil
import gzip
import io
from pooch import Unzip, retrieve
#import magic

Draschkow2018_URL = "https://zenodo.org/record/1421347/files/"

class Draschkow2018(BaseDataset):

    path_to_dataset=" "
    def __init__(self):
        super().__init__(
            subjects=list(range(1, 41)),
            sessions_per_subject=1,
            events=dict(Consistent=111, Inconsistent=112),
            code="draschkow 2018",
            interval=[-0.1, 0.9],
            paradigm="n400",
            doi="https://doi.org/10.5281/zenodo.3266929",
            dataset_path=None,
            )

    def fix_overlapping_events(self, events):
        """Add 1 to current event's sample time if it overlaps with the previous one"""
        old_t = -1000
        for ii, line in enumerate(events):
            new_t = line[0]
            if old_t == new_t:
                events[ii, 0] += 1
            old_t = new_t
        return events

    def _get_single_subject_data(self, subject):
        """return data for a single subject"""
         
        file_path_list = self.data_path(subject)
        sessions = {}
        session_name="session_1"
        run = 'run_1'
        if session_name not in sessions.keys():
            sessions[session_name] = {}
        
        raw = mne.io.read_raw_brainvision(file_path_list,
                                          preload=True, verbose=False, eog=['SO1', 'SO2'])
        raw.rename_channels({'FP1':'Fp1', 'FP2':'Fp2'})
        raw.set_montage("standard_1020")
        if(("M1" in raw.ch_names) | ("M2" in raw.ch_names)):
            #raw.drop_channels(["M1", "M2"])
            raw.set_eeg_reference(ref_channels=['M1', 'M2'])
        events, events_id=mne.events_from_annotations(raw, verbose=False)

        if events_id.get('Stimulus/111') == 10003:
            rows_congruent = np.where(events[:,2] == 10003)[0]
            events[rows_congruent,2] = 111
            rows_incongruent = np.where(events[:,2] == 10004)[0]
            events[rows_incongruent,2] = 112

        #events=self.fix_overlapping_events(events)
        #print("before fixing events events", len(np.where(events==111)[0]))
        # raw.set_annotations(mne.Annotations(onset=events[:, 0], duration=np.zeros(len(events)), 
        #                           description=events[:, 2]))

        # #events=self.fix_overlapping_events(events)
        # # events, _ = mne.events_from_annotations(
        # #             raw, event_id=event_id, verbose=False
        # #         )
        # events, events_id=mne.events_from_annotations(raw, verbose=False)
        # print("After fixing the events", len(np.where(events==111)[0]))
        #print("Raw annotations", raw.annotations)
        # raw.set_annotations(None)
        # desc = [str(e) for e in events[:, 2]]
        # onset = events[:, 0] / raw.info['sfreq']
        # duration = np.zeros(len(events))
        # annot = mne.Annotations(onset=onset, duration=duration, description=desc)
        # raw.set_annotations(annot)
        # events, events_id=mne.events_from_annotations(raw, verbose=False)
        # print("After fixing the events", len(np.where(events==10003)[0]))
        sessions[session_name][run]=raw, events
        return sessions
    
    def data_path(self, subject, path=None, force_update=False,
                  update_path=None, verbose=None): 
        """
        The subjects which does not have M1 and M2 channels in their dataset. 
        It is mainly because an reference (M1+M2) is used instead of channels 
        M1 and M2 and Stimulus/111 is marked 10003 and simulus/112 is marked 
        10004 for such kind of event_ids. "Stimuli/101 and Stimuli/102"--> total no. of trails i.e. 169, 
        mostly, it depicts the start or end of very trail. 

        Stimuli/111 and Stimuli/112" --> Congruent and Incongruent
        Stimuli/121 and Stimuli/122" --> 17 reptitive trails consisting of congruent and incongruent
        Stimuli/S111 and Stimuli/111" --> both belongs to congruent with even codes 111 and 10003
        Stimuli/S112 and Stimuli/112" --> both belongs to Incongruent with even codes 112 and 10004

        """   
        if subject not in self.subject_list:
            raise ValueError("Invalid subject number")

        # define url and paths
        base_url = Draschkow2018_URL
        #subject_str = f"subject_{subject:02}"
        url = f"{base_url}n400Data.zip"
        zip_filename = f"n400Data.zip"
        main_directory='n3n4'
        if(Draschkow2018.path_to_dataset==" "):
            path_zip = dl.data_dl(url, "Draschkow2018")
            Draschkow2018.path_to_dataset=path_zip
        else:
            path_zip=Draschkow2018.path_to_dataset

        self.dataset_path=os.path.dirname(os.path.dirname(Path(path_zip.strip(zip_filename))))   
        #other_dataset_zip="Raw_EEG_Data.zip"
        print("path zip", path_zip)
        subject_dir = Path(path_zip.strip(zip_filename))/main_directory
        #other_dataset_path=Path(path_zip.strip(zip_filename))/other_dataset_zip
        if not subject_dir.exists():
            with z.ZipFile(subject_dir, "r") as zip_ref:
                zip_ref.extractall(subject_dir)

        subject_dict=OrderedDict()
        unprocessed_data_path=os.path.join(subject_dir, "unprocessed")
        subject_dir = os.listdir(unprocessed_data_path)
        i=0
        for sub in subject_dir:
            if((sub.split('.')[1]=='vhdr')&(sub.split('_')[1]=='vis')):
                subject_dict[self.subject_list[i]]=os.path.join(unprocessed_data_path,sub)
                i=i+1
        return subject_dict.get(int(subject))



        # import zipfile

        # zip_file = zipfile.ZipFile(file_name, 'r')
        # zip_file.extractall()
        # zip_file.close()

        

        #print("s")
        # print("subject directory:", subject_dir)
        # print("===================================================================")
        # print("===================================================================")
        # #mime = magic.Magic(mime=True)
        # #mimetype = mime.from_file(path_zip)
        # #print(mimetype)
        # if not subject_dir.exists():
        #     print("I am inside zip file")
        #     os.mkdir(subject_dir)
        #     retrieve(
        #         FILES[ind],
        #         None,
        #         dataname + ".zip",
        #         base_path,
        #         processor=Unzip(),
        #         progressbar=True,
        #     )
            #zip_ref=z.read
            #with open(path_zip, 'rb') as f:
             #   zip=z.ZipFile(io.BytesIO(f))
              #  zip.extractall(subject_dir)  
            #f=gzip.open(path_zip,'rb')
            #file_content=f.read()
            #print(file_content)
            #print

            
            #shutil.unpack_archive(path_zip, subject_dir)
        #     with zipfile.ZipFile(path_zip, 'r') as zip_ref:
        #         # check if the zip file is valid
        #         if not zip_ref.testzip():
        #             # extract all files to the specified directory
        #             zip_ref.extractall(subject_dir)
        #         else:
        #             print('Zip file is corrupt or incomplete.')

            #patoolib.extract_archive(path_zip, outdir=subject_dir)
            #Archive(path_zip).extractall(subject_dir)

            #print("Whether file is zip or not: ", z.is_zipfile(path_zip))
        # with open(path_zip, 'rb') as f:
        #     first_bytes = f.read(4)
        #     print(first_bytes)
            #f.extractall(subject_dir)
            #zip_ref = z.ZipFile(path_zip, "rb")
            #print(zip_ref.read(4))
            #print(zip_ref.read(4))
            
            #with z.ZipFile(path_zip, "r") as zip_ref:
            #with z.GzipFile(path_zip) as Zip:
             #   for ZipMember in Zip.infolist():
              #      Zip.extract(ZipMember, path=subject_dir)
                #zip_ref.extractall(subject_dir)
                                
if __name__ == "__main__":
    d=Draschkow2018()
   #bi15a.download()
    print(d.get_data())

    #print(g._get_single_subject_data(2))


