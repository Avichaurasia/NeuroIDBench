### Example 1: 
Benchmarking pipeline using the dataset’s default parameters and auto-regressive features with SVM classification

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: brainModels.BrainInvaders2015a
    from: datasets

pipelines:

  "AR+PSD+SVM": 
    - name: AutoRegressive
      from: brainModels.featureExtraction

    - name: SVC
      from: sklearn.svm
      parameters: 
        kernel: 'rbf'
        class_weight: "balanced"
        probability: True
```

This benchmarking pipeline consists of default dataset's parameters which means EEG data of all the subjects are utlized in this pipeline.
Further, default epochs interval is set [-0.2, 0.8], rejection_threshold is None. AR features with default order 6 and SVM with kernel 'rbf',
constitutes the pipeline. 

### Example 2: 
Benchmarking pipeline by setting parmeters for the dataset, AR features with AR order 6 and algorithm SVM with kernel 'rbf'

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: brainModels.BrainInvaders2015a
    from: datasets
    parameters:
        subjects: 10
        interval: [-0.1, 0.9] 
        rejection_threshold: 200

  pipelines: 

  "AR+SVM":
    - name: AutoRegressive 
      from: brainModels.featureExtraction 
      parameters:
        order: 5

    - name: SVC
      from: sklearn.svm 
      parameters:
        kernel: ’rbf’ 
        class_weight: "balanced" 
        probability: True
```

This benchmarking pipeline first set parameters for the datasets such as EEG data of only 10 subjects will be utlized, epochs rejection threshol
is set 200 microvolts for dropping aritifcats. AR features with order 5 and SVM with kernel 'rbf' depicts the parameters for the feature extraction
and authentication algorithm. 

### Example 3: 
Benchamrking pipeline for dataset BrainInvaders15a with AR and PSD features with classifier SVM. 

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: brainModels.BrainInvaders2015a
    from: datasets
    parameters:
        subjects: 10
        interval: [-0.1, 0.9] 
        rejection_threshold: 200


  pipelines: 
  
  "AR+SVM":
    - name: AutoRegressive 
      from: brainModels.featureExtraction
      parameters:
        order: 5

    - name: PowerSpectralDensity 
      from: brainModels.featureExtraction

    - name: SVC
      from: sklearn.svm 
      parameters:
        kernel: ’rbf’ 
        class_weight: "balanced" 
        probability: True
```

This benchmarking pipeline first set parameters for the datasets such as EEG data of only 10 subjects will be utlized, epochs rejection threshol
is set 200 microvolts for dropping aritifcats. AR and PSD features with SVM is utilized to form the sklearn pipeline. 

### Example 4: 
Benchamrking pipeline for dataset BrainInvaders15a with Siamese Networks

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: brainModels.BrainInvaders2015a
    from: datasets
    parameters:
        subjects: 10
        interval: [-0.1, 0.9] 
        rejection_threshold: 200

  pipelines:

  "TNN":
    - name : TwinNeuralNetwork
    from: brainModels.featureExtraction
    parameters:
        EPOCHS: 100 
        batch_size: 192 
        verbose: 1 
        workers: 1  
```

This benchmarking pipeline first set parameters for the datasets such as EEG data of only 10 subjects will be utlized, epochs rejection threshol
is set 200 microvolts for dropping aritifcats. Here, Siamese neural Network pipeline with parameters such EPOCHS=10, batch_size=256 is set for 
training the neural network.

### Example 5: 
Benchamrking pipeline for dataset BrainInvaders15a with traditional and deep learning methods

```bash
name: "BrainInvaders2015a"

dataset: 
  - name: brainModels.BrainInvaders2015a
    from: datasets
    parameters:
        subjects: 10
        interval: [-0.1, 0.9] 
        rejection_threshold: 200

  pipelines:

   "AR+SVM":
    - name: AutoRegressive 
      from: brainModels.featureExtraction 
      parameters:
        order: 5

    - name: PowerSpectralDensity 
      from: brainModels.featureExtraction

    - name: SVC
      from: sklearn.svm 
      parameters:
        kernel: ’rbf’ 
        class_weight: "balanced" 
        probability: True

   "TNN":
    - name : TwinNeuralNetwork
    from: featureExtraction
    parameters:
        EPOCHS: 10 
        batch_size: 256 
        verbose: 1 
        workers: 1  
```

This benchmarking pipeline first set parameters for the datasets such as EEG data of only 10 subjects will be utlized, epochs rejection threshol
is set 200 microvolts for dropping aritifcats. Here, the pipeline consisiting of traditional algorithm such as SVM and deep learning method like Siamese Neural Networks is made.  

Launch the automation Script: 

Launch the python file run.py with the following command. 
```bash
python brainModels/run.py
```











