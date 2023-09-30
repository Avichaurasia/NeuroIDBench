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
    LeaveOneGroupOut
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

# This function has been sourced from https://git.scc.kit.edu/ps-chair/brainnet licensed under the Creative Commons
def _predict_close_set(embedding_network, x_train_val, y_train_val, x_test, y_test):
    """Calculates similarity values for closed-set recognition by comparing each face embedding in the test set with all face embeddings 
    in the training set, and then calculating the euclidean distance between each pair of embeddings. It then uses these similarity 
    values to create pairs of faces, with the label 1 indicating that the faces belong to the same individual (from the training set) 
    and label 0 indicating that the faces belong to different individuals (including unknown identities)"""

    resutls=[]
    resutls2=[]
    resutls3=defaultdict(list)
    calsstrain=np.unique(y_train_val)
    TP,FP,TN,FN=0,0,0,0
    digit_indices = [np.where(y_train_val == i)[0] for i in np.unique(y_train_val)]
    x_test_1 = x_test

    print(len(x_train_val),len(x_test))
    anc_e=embedding_network(x_test[0:min(500,len(x_test))])
    for c in range(len(x_test)//500):
        anc_e=tf.concat(axis=0, values = [anc_e, embedding_network(x_test[(c+1)*500:min((c+2)*500,len(x_test))])]) 	
    anc_et=embedding_network(x_train_val[0:min(500,len(x_train_val))])
    for c in range(len(x_train_val)//500):
        anc_et=tf.concat(axis=0, values = [anc_et, embedding_network(x_train_val[(c+1)*500:min((c+2)*500,len(x_train_val))])]) 
    print(len(anc_et),len(anc_e))
    for i in tqdm(range(len(x_test_1)), desc="Calculating similarity"):
        prediction=[]
        test_e=embedding_network(np.array([x_test[i]]))
        same_in=digit_indices[np.where(calsstrain == y_train_val[i])[0][0]]
        
        for t in range(len(x_train_val)):
            tempp=-1*euclidean_distance2(anc_et[t],test_e).numpy()[0][0] 
            if y_test[i] ==y_train_val[t]:
                resutls.append([tempp,1,y_test[i],y_train_val[t]])
            else:
                resutls.append([tempp,0,y_test[i],y_train_val[t]])

            prediction.append(tempp)        
        prediction=np.array(prediction)
        
        for j in calsstrain:
            same_in=digit_indices[np.where(calsstrain == j)[0][0]]
            spredict=((sum(prediction[same_in]))/(len(same_in)))            
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
    
    def _close_set(self, data, y, groups, siamese):
        count_cv=0
        dicr3={}
        dicr2={}
        dicr1={}
        #cv=LeaveOneGroupOut()
        
        return (dicr1, dicr2, dicr3)
    
    def _open_set(self, X, y, groups, siamese):
        dicr3={}
        dicr2={}
        dicr1={}
        x_train=[]
        y_train=[]
        x_test=[]
        y_test=[]
        session_id=np.unique(groups)
        # Divide the x and y into train and test based on session_id such that session_id=1 is train and session_id=2,3 is for test
        for i in range(len(session_id)):
            if i==0:
                x_train=X[groups==session_id[i]]
                y_train=y[groups==session_id[i]]
            else:
                x_test=X[groups==session_id[i]]
                y_test=y[groups==session_id[i]]
        # Train the siamese network on the train data

        
        
        return (dicr1, dicr2, dicr3)
        
    def _evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")
        X, _, metadata=self.paradigm.get_data(dataset)
        results_saving_path=os.path.join(
            dataset.dataset_path,
            "Results",
            "SiameseCrossSessionEvaluation"
            #f"{dataset.code}_CloseSetEvaluation")
        )
        if not os.path.exists(results_saving_path):
            os.makedirs(results_saving_path)

        if(dataset.paradigm == "p300"):
            metadata=metadata[metadata['event_id']=="Target"]
            
        elif (dataset.paradigm == "n400"):
            metadata=metadata[metadata['event_id']=="Inconsistent"]

       # print("subjects and sessions after selection", metadata[['subject', 'session']].value_counts())

        metadata=self._valid_subject(metadata, dataset)
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
                close_dicr1, close_dicr2, close_dicr3=self._close_set(data, y, groups, siamese)
                    
                    

            if self.return_open_set:
                open_dicr1, open_dicr2, open_dicr3=self._open_set(data, y, groups, siamese)  

                   
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

    