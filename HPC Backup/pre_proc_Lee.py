def get_data_old(self, dataset, subjects=None, return_epochs=False):
        if not self.is_valid(dataset):
            message = f"Dataset {dataset.code} is not valid for paradigm"
            raise AssertionError(message)
        replacement_dict = {v: k for k, v in dataset.event_id.items()}

        # This returns the raw mne data for the given number of subjects in the form of dictionary
        data = dataset.get_data(dataset.subject_list)
        self.prepare_process(dataset)
        epochs_directory=os.path.join(dataset.dataset_path, "Epochs")
        print("epochs directory", epochs_directory)
        if not os.path.exists(epochs_directory):
            os.makedirs(epochs_directory)
        else:
            print("Epochs folders already created!")
        X = []
        labels = []
        metadata = []
        subject_dict=OrderedDict()
        for subject, sessions in tqdm(data.items(), desc="Extracting epochs"):
            subject_directory=os.path.join(epochs_directory,str(subject))
            if not os.path.exists(subject_directory):
                os.makedirs(subject_directory)
            subject_dict[subject]={}

            for session, runs in sessions.items():
                session_directory=os.path.join(subject_directory, session)
                if not os.path.exists(session_directory):
                    os.makedirs(session_directory)
                subject_dict[subject][session]={}

                for run, raw_events in runs.items():
                    raw=raw_events[0]
                    events=raw_events[1]
                    subject_dict[subject][session][run]={} 


                    # proc = self.process_raw(raw, events, dataset, return_epochs)
                    # x, lbs = proc
                    # if (proc is None) or (len(x)==0):
                    # # this mean the run did not contain any selected event
                    # # go to next
                    #     continue
                    # subject_dict[subject][session][run]=x
                    # X.append(x)
                    # labels = np.append(labels, lbs, axis=0)


                    pre_processed_epochs=os.path.join(session_directory, f"{run}_epochs.fif")

                    #if return_epochs:
                    if not os.path.exists(pre_processed_epochs):
                        
                        proc = self.process_raw(raw, events, dataset, return_epochs)
                        x, lbs = proc
                        if (proc is None) or (len(x)==0):
                        # this mean the run did not contain any selected event
                        # go to next
                            continue
                        x.save(pre_processed_epochs, overwrite=True)
                        subject_dict[subject][session][run]=x
                        X.append(x)
                        labels = np.append(labels, lbs, axis=0)
                        
                        #del proc
                    else:
                        # print("target events", len(np.where(events[:,2]==1)[0]))
                        # print("non-target events", len(np.where(events[:,2]==2)[0]))
                        x=mne.read_epochs(pre_processed_epochs, preload=True, verbose=False)
                        subject_dict[subject][session][run]=x
                        X.append(x)
                    met = pd.DataFrame(index=range(len(x)))
                    met["subject"] = subject
                    met["session"] = session
                    met["run"] = run
                    met["event_id"] = x.events[:, 2].astype(int).tolist()
                    met["event_id"]=met["event_id"].map(replacement_dict)
                    metadata.append(met)
        metadata = pd.concat(metadata, ignore_index=True)
        if return_epochs:
            X = mne.concatenate_epochs(X, verbose=False)
            return X, subject_dict, metadata  
        else:
            X = mne.concatenate_epochs(X, verbose=False).get_data()
            return X, subject_dict, metadata
        

def lee_get_data(self, dataset, subjects=None, return_epochs=False):
    if not self.is_valid(dataset):
        message = f"Dataset {dataset.code} is not valid for paradigm"
        raise AssertionError(message)
    replacement_dict = {v: k for k, v in dataset.event_id.items()}

    # This returns the raw mne data for the given number of subjects in the form of dictionary
    epochs_directory=os.path.join(dataset.dataset_path, "Epochs")
    X = []
    labels = []
    metadata = []
    subject_dict=OrderedDict()
    subject_list=os.listdir(epochs_directory)
    for subject in tqdm(subject_list, desc="Extracting epochs"):
        subject_dir=Path(os.path.join(epochs_directory, subject))
        session_list=os.listdir(os.path.join(epochs_directory, subject))
        subject_id=int(subject)
        subject_dict[subject_id]={}
        for sessions in session_list:
            session_dir=Path(os.path.join(subject_dir, sessions))
            run_list=os.listdir(os.path.join(subject_dir, sessions))
            subject_dict[subject_id][sessions]={}
            for run in run_list:
                run_dir=Path(os.path.join(session_dir, run))
                x=mne.read_epochs(run_dir, preload=True, verbose=False)
                print(x)
                subject_dict[subject_id][sessions][run]=x
                X.append(x)

                met = pd.DataFrame(index=range(len(x)))
                met["subject"] = subject_id
                met["session"] = sessions
                met["run"] = run
                met["event_id"] = x.events[:, 2].astype(int).tolist()
                met["event_id"]=met["event_id"].map(replacement_dict)
                metadata.append(met)

    metadata = pd.concat(metadata, ignore_index=True)
    X = mne.concatenate_epochs(X, verbose=False).get_data()
    return X, subject_dict, metadata 