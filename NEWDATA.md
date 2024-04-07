## Add new EEG data

Reserachers can utilize this tool to perform benchmarking and evalaute their approach. 
However,there are certain pre-requisities that need to be fuflfilled to add new EEG data to this tool.
Following steps needs to be followed to add new EEG data.

1. Convert the raw EEG data into standarized MNE raw object(mne.io.Raw). MNE is a powerful Python package and capable to reading and 
    converting any EEG format into MNE format. Some of the tutorials for converting EEG data into standarized MNE format
    can be found at https://mne.tools/stable/auto_tutorials/io/index.html. 

2. Once the raw EEG data is converted into mne. Save the state of MNE object into .fif format. 
    The mne data should be saved in hierarchy of folders like "sujectID"--> "Sesssion number" --> "Run number" --> EEG_data.fif.
    For example: Assume an EEG dataset comprises of 2 subjects. Each subject has performed EEG task across 2 sessions
    and each session contains two runs. First create a folder with name <b>User_data</b> on your local system. This folder should contain the MNE data of users. Then the mne data should be saved inside <b>User_data</b> folder in the following way: 

    <b>Subject 1</b>: 
    "Subject_1" --> "Session_1" --> "Run_1" --> EEG_data.fif, 
    "Subject_1" --> "Session_2" --> EEG_data.fif 

    <b>Subject</b>:
    "Subject_2" --> "Session_1" --> "Run_1" --> EEG_data.fif, 
    "Subject_2" --> "Session_2" --> "Run_1" --> EEG_data.fif 

3. Edit the single_dataset.yml with the below configurations:

Benchamrking pipeline for User i.e., Reseracher's own MNE data with traditional and deep learning methods

```bash
name: "User"

dataset: 
  - name: UserDataset
    from: neuroIDBench.datasets
    parameters: 
      dataset_path: '<local_system_to_folder_User_data>'
    
pipelines:

  "AR+PSD+SVM": 
    - name: AutoRegressive
      from: neuroIDBench.featureExtraction
      parameters: 
        order: 6
        
    - name: PowerSpectralDensity
      from: neuroIDBench.featureExtraction
        
    - name: SVC
      from: sklearn.svm
      parameters: 
        kernel: 'rbf'
        class_weight: "balanced"
        probability: True

  "TNN": 
    - name : TwinNeuralNetwork
      from: neuroIDBench.featureExtraction
      parameters: 
        EPOCHS: 10
        batch_size: 256
        verbose: 1
        workers: 1

  
  "AR+PSD+RF": 
  - name: AutoRegressive
    from: neuroIDBench.featureExtraction
    parameters: 
      order: 6
    
  - name: PowerSpectralDensity
    from: neuroIDBench.featureExtraction
      
  - name: RandomForestClassifier
    from: sklearn.ensemble
    parameters: 
        class_weight: "balanced"
```

This benchmarking pipeline reads the MNE data from the folder User_data and create a dataset instance. 
Afterwards, the pipeline consisiting of traditional algorithm such as SVM and  deep learning method 
like Siamese Neural Networks is made.  

4. Launch the python file run.py from terminal which has a main method and internally calls the automation script for benchmark.py 