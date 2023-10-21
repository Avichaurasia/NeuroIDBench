import logging
import os
from copy import deepcopy
from time import time
from typing import Optional, Union
import joblib
import numpy as np
from mne.epochs import BaseEpochs
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    GroupKFold,
    LeaveOneGroupOut,
    train_test_split
    ,
)
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from deeb.Evaluation.base import BaseEvaluation
#from deeb.evaluation.scores import Scores as score
from collections import OrderedDict
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity as cs
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from deeb.pipelines.siamese import Siamese
from keras.callbacks import EarlyStopping
from collections import defaultdict
from deeb.Evaluation.scores import Scores as score
import pickle
import mne
from tabnanny import verbose
from tqdm import tqdm 
log = logging.getLogger(__name__)

Vector = Union[list, tuple, np.ndarray]

# This function has been sourced from https://git.scc.kit.edu/ps-chair/brainnet licensed under the Creative Commons
def _predict_open_set(embedding_network, x_test, y_test):
    """Calculates similarity values for authentication (verification) by 
    comparing each face embedding in the test set with all other face embeddings, and then calculating the cosine similarity between 
    each pair of embeddings. It then uses these similarity values to create pairs of faces, with the label 1 indicating that the faces
    belong to the same individual and label 0 indicating that the faces belong to different individuals."""

    # Compute embeddings for all test samples
    resutls=[]
    resutls2=[]
    resutls3=defaultdict(list)
    pair1=[]
    pair2=[]
    calss=np.unique(y_test)
    TP,FP,TN,FN=0,0,0,0
    digit_indices = [np.where(y_test == i)[0] for i in np.unique(y_test)]
    x_test_1 = x_test
    print(len(x_test))
    anc_e=embedding_network(x_test[0:min(500,len(x_test))])
    for c in tqdm(range(len(x_test)//500), desc='Getting test embedings'):
        anc_e=tf.concat(axis=0, values = [anc_e, embedding_network(x_test[(c+1)*500:min((c+2)*500,len(x_test))])]) 	
    print(len(x_test))
    for i in tqdm(range(len(x_test_1)), desc="Calculating similarity"):
        temp=np.where(calss == y_test[i])[0][0]
        prediction=[]
        same_in=digit_indices[np.where(calss == y_test[i])[0][0]]
        for t in range(len(x_test_1)):
            tempp=-1*euclidean_distance2(anc_e[t],anc_e[i]).numpy()[0]
            if t in same_in:
                if t==i:
                    pass
                else:
                    resutls.append([tempp,1,y_test[i],y_test[t]])
            else:
                resutls.append([tempp,0,y_test[i],y_test[t]])    
            prediction.append(tempp)
        prediction=np.array(prediction)
        
        for j in calss:
            same_in=digit_indices[np.where(calss == j)[0][0]]
            same_in=np.setdiff1d(same_in,[i])
            spredict=max(prediction[same_in])        
    
            if y_test[i] ==j:
                resutls2.append([spredict,1,y_test[i],j])
                resutls3[j].append([spredict,1,y_test[i],j])
            else:
                resutls2.append([spredict,0,y_test[i],j])
                resutls3[j].append([spredict,0,y_test[i],j])
                    
            if spredict>0.85:
                if y_test[i] ==j:
                    TP+=1
                else:
                    FP+=1
            else:
                if y_test[i] == j:
                    FN+=1
                else:
                    TN+=1
    return resutls,resutls2,resutls3
            
def euclidean_distance2(x, y):
	sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=None, keepdims=True)
	return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))
    
# This needs to be checked later and modified
class Siamese_CrossSessionEvaluation(BaseEvaluation):
    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        return_close_set: bool = True,
        return_open_set: bool = True,
        **kwargs
    ):
        self.n_perms = n_perms
        self.data_size = data_size
        self.return_close_set = return_close_set
        self.return_open_set = return_open_set
        super().__init__(**kwargs)
     
    def _open_set(self, X, y, groups, siamese):
        count_session=2
        dicr3={}
        dicr2={}
        dicr1={}
        
        # Gettin the unique session IDs
        unique_sessions=np.unique(groups)

        # Find subjects from Session 1
        session_1_subject_indices = np.where(groups == 'session_1')[0]
        X_train=X[session_1_subject_indices]
        y_train=y[session_1_subject_indices]
        
        # Find all the subject IDs from sessions
        session_1_subject_ids=np.unique(y[session_1_subject_indices])
          
        # Randomise the subject ids from session 1
        np.random.shuffle(session_1_subject_ids) 

        # Randomly select 75% of subjects from Session 1 for training 
        no_subjects_for_train = int(np.ceil(0.75 * len(session_1_subject_ids)))
        train_subjects = session_1_subject_ids[0:no_subjects_for_train]
        train_indices = np.where(np.isin(y_train, train_subjects))
        test_subjectsIds=np.setdiff1d(session_1_subject_ids, train_subjects)

        # Getting the training data
        X_train=X_train[train_indices]
        y_train=y_train[train_indices]
        scaler = StandardScaler()

        # Normalising the training data
        X_train = scaler.fit_transform(X_train.reshape((X_train.shape[0], -1))).reshape(X_train.shape) 
        tf.keras.backend.clear_session()
        model=siamese._siamese_embeddings(X_train.shape[1], X_train.shape[2])
        embedding_network=model
        early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(siamese.batch_size)
        history = embedding_network.fit(train_dataset,
                                        workers=siamese.workers,
                                        epochs=siamese.EPOCHS,
                                        verbose=siamese.verbose)

        # Loop through other sessions for testing
        for session_id in unique_sessions:

            # Skip session 1 as it is used for training
            if session_id != 'session_1':
                session_subject_indices = np.where(groups == session_id)[0] 
                X_test=X[session_subject_indices]
                y_test=y[session_subject_indices]

                # Remove the subjects from session 1 from the test data
                test_indices=np.where(np.isin(y_test, test_subjectsIds))

                # Getting the test data for subjects not used for training from each sessions except session 1
                X_test=X_test[test_indices]
                y_test=y_test[test_indices]
                X_test = scaler.transform(X_test.reshape((X_test.shape[0], -1))).reshape(X_test.shape)

                # Getting the similarity scores from session except session 1 under open set scenario
                resutls1,resutls2,resutls3=_predict_open_set(model, X_test, y_test) 
                dicr1[count_session] = resutls1
                dicr2[count_session] = resutls2
                dicr3.update(dict(resutls3))
                count_session=count_session+1  
        return (dicr1, dicr2, dicr3)
        
    def _evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
        
        # Getting the EEG data from the dataset
        X, _, metadata=self.paradigm.get_data(dataset)
        results_saving_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "SiameseCrossSessionEvaluation"
        )
        if not os.path.exists(results_saving_path):
            os.makedirs(results_saving_path)

        if(dataset.paradigm == "p300"):
            metadata=metadata[metadata['event_id']=="Target"]
            
        elif (dataset.paradigm == "n400"):
            metadata=metadata[metadata['event_id']=="Inconsistent"]

        metadata=self._valid_subject(metadata, dataset)
        #print("session numbers", metadata['session'].value_counts())
        target_index=metadata['event_id'].index.tolist()
        data=X[target_index]
       
        # Selecting the subject labels for the target or inconsistent trails
        y=np.array(metadata["subject"])
        results_close_set=[]
        results_open_set=[]
        groups = metadata.session.values
        for name, clf in pipelines.items(): 
            siamese = clf[0]
            if self.return_close_set:
                raise AssertionError("Close-set is not allowed for cross-session evaluation")
                    
            if self.return_open_set:
                open_dicr1, open_dicr2, open_dicr3=self._open_set(data, y, groups, siamese) 
                open_set_path=os.path.join(results_saving_path,"open_set")
                if not os.path.exists(open_set_path):
                    os.makedirs(open_set_path)
                
                with open(os.path.join(open_set_path, "d1_dicr1.pkl"), 'wb') as f:
                    pickle.dump(open_dicr1, f)

                with open(os.path.join(open_set_path, "d1_dicr2.pkl"), 'wb') as f:
                    pickle.dump(open_dicr2, f)

                with open(os.path.join(open_set_path, "d1_dicr3.pkl"), 'wb') as f:
                    pickle.dump(open_dicr3, f)

                for sub in open_dicr3.keys():
                    results=np.array(open_dicr3[sub])
                    true_lables=np.array(results[:,1])
                    predicted_scores=np.array(results[:,0])
                    inter_tpr, auc, eer, frr_1_far=score._calculate_siamese_scores(true_lables, predicted_scores)
                    res_open_set = {
                    'evaluation': 'Cross Session',
                        "eval Type": "Open Set",
                        "dataset": dataset.code,
                        "pipeline": name,
                        "subject": sub,
                        #"session": session,
                        "frr_1_far": frr_1_far,
                        #"accuracy": mean_accuracy,
                        "auc": auc,
                        "eer": eer,
                        "tpr": inter_tpr,
                        #"std_auc": std_auc,
                        "n_samples": len(X)  # not training sample
                        #"n_channels": data.columns.size
                        }
                    results_open_set.append(res_open_set)   

        scenario='open_set'
        return results_open_set, scenario
                   
    def evaluate(self, dataset, pipelines, param_grid):
        results, scenario=self._evaluate(dataset, pipelines)

        results_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "SiameseCrossSessionEvaluation"
        )
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        return results, results_path, scenario
    
    def _valid_subject(self , metadata, dataset):
        subject_sessions = metadata.groupby('subject')['session'].nunique()
        valid_subjects = subject_sessions[subject_sessions == dataset.n_sessions].index
        metadata = metadata[metadata['subject'].isin(valid_subjects)]      
        return metadata
 
    
    def is_valid(self, dataset):
        return dataset.n_sessions > 1

    