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
from deeb.pipelines.siamese import Siamese
from keras.callbacks import EarlyStopping
log = logging.getLogger(__name__)

Vector = Union[list, tuple, np.ndarray]

class Siamese_WithinSessionEvaluation(BaseEvaluation):
    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        # dataset=None,
        return_close_set: bool = True,
        return_open_set: bool = True,
        # paradigm=None,
        #paradigm=None,
        **kwargs
    ):
        # self.dataset = dataset
        # self.paradigm = paradigm
        #self.paradigm = paradigm
        self.n_perms = n_perms
        self.data_size = data_size
        self.return_close_set = return_close_set
        self.return_open_set = return_open_set
        super().__init__(**kwargs)

    def _predict(self, embedding_network, x_test, y_test):
        # Compute embeddings for all test samples

        results = []
        results_by_class = defaultdict(list)
        labels = np.unique(y_test)
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
        digit_indices = [np.where(y_test == i)[0] for i in labels]
        
        # Compute embeddings for all test samples
        test_embeddings = []
        for c in range(0, len(x_test), 500):
            embeddings_chunk = embedding_network(x_test[c:c+500])
            test_embeddings.append(embeddings_chunk)
        test_embeddings = tf.concat(test_embeddings, axis=0)
        
        # Compute cosine similarity between all pairs of embeddings
        pairwise_similarities = cs(test_embeddings, test_embeddings)
        
        for i in range(len(x_test)):
            # Compute scores for all other samples
            scores = pairwise_similarities[i]
            same_in = digit_indices[np.where(labels == y_test[i])[0][0]]
            
            for j in range(len(x_test)):
                score = scores[j]
                label_i, label_j = y_test[i], y_test[j]
                is_same = int(j in same_in)
                results.append([score, is_same, label_i, label_j])
                
                # Update results by class
                results_by_class[label_j].append([score, is_same, label_i, label_j])
                
            # Compute scores for same-class samples
            same_scores = [pairwise_similarities[i][j] for j in same_in if j != i]
            max_same_score = max(same_scores)
            
            for label_j in labels:
                is_same = int(label_j == label_i)
                scores_j = [pairwise_similarities[j][k] for j in digit_indices[np.where(labels == label_j)[0][0]] for k in digit_indices[np.where(labels == label_j)[0][0]] if j != i]
                max_other_score = max(scores_j)
                threshold = 0.85
                
                if label_i == label_j:
                    if max_same_score > threshold:
                        true_positive = true_positive+1
                    else:
                        false_negative = false_negative+1
                else:
                    if max_other_score > threshold:
                        false_positive = false_positive+1
                    else:
                        true_negative = true_negative+1
                
                results_by_class[label_j].append([max_same_score, is_same, label_i, label_j])
                
        return results, results_by_class
    
    def _close_set(self, data, y, grid_clf):
        
        #print("Close Set")
        # Defining the Stratified KFold
        skfold = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
        for train_index, test_index in skfold.split(data, y):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = y[train_index], y[test_index]
            tf.keras.backend.clear_session()
            #grid_clf = clone(clf)
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(500).batch(128)
            #print("classifer", grid_clf)
            
            model=grid_clf._siamese_embeddings(X_train.shape[1], X_train.shape[2])
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
            history = model.fit(train_dataset,
                                        workers=5,
                                        callbacks=[early_stopping_callback])

            results, results_by_class=self._predict(model, X_test, y_test) 
            #results, results_by_class=self._predict(model, X_test, y_test) 
    
    def _open_set(self, data, y, grid_clf):
        #for name, clf in pipelines.items():
        groupfold = GroupKFold(n_splits=4,shuffle=True,random_state=42)
        for train_index, test_index in groupfold.split(data, y, groups=y):
            X_train, X_test = data[train_index], data[test_index]
            y_train, y_test = y[train_index], y[test_index]
            tf.keras.backend.clear_session()
            #grid_clf = clone(clf)
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(128)
            #print("classifer", grid_clf)
            model=grid_clf._siamese_embeddings(X_train.shape[1], X_train.shape[2])
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
            history = model.fit(train_dataset,
                                        workers=5,
                                        callbacks=[early_stopping_callback])

            results, results_by_class=self._predict(model, X_test, y_test) 
        return 0
 
    def evaluate(self, dataset, pipelines, param_grid):
        y=[]
        X, _, metadata=self.paradigm.get_data(dataset)
        if(dataset.paradigm == "p300"):
            target_index=metadata[metadata['event_id']=="Target"].index.tolist()
            metadata=metadata[metadata['event_id']=="Target"]

        elif (dataset.paradigm == "n400"):
            target_index=metadata[metadata['event_id']=="Inconsistent"].index.tolist()
            metadata=metadata[metadata['event_id']=="Inconsistent"]
        
        # Selecting the target trials if paradigm is p300 or inconsistent if paradigm is n400
        data=X[target_index]

        # Selecting the subject labels for the target or inconsistent trails
        #y=metadata.iloc[target_index]["subject"].tolist()
        y=np.array(metadata["subject"])
        #y-np.array(y)
        #print("Y: ", type(y), y)
        # iterate over sessions
        results_close_set=[]
        results_open_set=[]
        for session in np.unique(metadata.session):
            ix = metadata.session == session
            for name, clf in pipelines.items():
                grid_clf = clone(clf)
                le = LabelEncoder()
                X_=data[ix]
                y_=le.fit_transform(y[ix])
                if self.return_close_set:
                    close_set_scores=self._close_set(X_, y_, grid_clf)
                    results_close_set.append(close_set_scores)

                elif self.return_open_set:
                    open_set_scores=self._open_set(X_, y_, grid_clf)   
                    results_open_set.append(open_set_scores)  

        if results_close_set:
            if (len(results_close_set) == 1):
                results_close_set = results_close_set[0]
            else:
                results_close_set = np.mean(results_close_set, axis=0)

        if results_open_set:
            if (len(results_open_set) == 1):
                results_open_set = results_open_set[0]
            else:
                results_open_set = np.mean(results_open_set, axis=0)

        return results_close_set, results_open_set
    
    def is_valid(self, dataset):
        return True
    
# This needs to be checked later and modified
class Siamese_CrossSessionEvaluation(BaseEvaluation):
    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        # dataset=None,
        return_close_set: bool = True,
        return_open_set: bool = True,
        # paradigm=None,
        #paradigm=None,
        **kwargs
    ):
        # self.dataset = dataset
        # self.paradigm = paradigm
        #self.paradigm = paradigm
        self.n_perms = n_perms
        self.data_size = data_size
        self.return_close_set = return_close_set
        self.return_open_set = return_open_set
        super().__init__(**kwargs)

    def _predict(self, embedding_network, x_test, y_test):
        # Compute embeddings for all test samples

        results = []
        results_by_class = defaultdict(list)
        labels = np.unique(y_test)
        true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
        digit_indices = [np.where(y_test == i)[0] for i in labels]
        
        # Compute embeddings for all test samples
        test_embeddings = []
        for c in range(0, len(x_test), 500):
            embeddings_chunk = embedding_network(x_test[c:c+500])
            test_embeddings.append(embeddings_chunk)
        test_embeddings = tf.concat(test_embeddings, axis=0)
        
        # Compute cosine similarity between all pairs of embeddings
        pairwise_similarities = cs(test_embeddings, test_embeddings)
        
        for i in range(len(x_test)):
            # Compute scores for all other samples
            scores = pairwise_similarities[i]
            same_in = digit_indices[np.where(labels == y_test[i])[0][0]]
            
            for j in range(len(x_test)):
                score = scores[j]
                label_i, label_j = y_test[i], y_test[j]
                is_same = int(j in same_in)
                results.append([score, is_same, label_i, label_j])
                
                # Update results by class
                results_by_class[label_j].append([score, is_same, label_i, label_j])
                
            # Compute scores for same-class samples
            same_scores = [pairwise_similarities[i][j] for j in same_in if j != i]
            max_same_score = max(same_scores)
            
            for label_j in labels:
                is_same = int(label_j == label_i)
                scores_j = [pairwise_similarities[j][k] for j in digit_indices[np.where(labels == label_j)[0][0]] for k in digit_indices[np.where(labels == label_j)[0][0]] if j != i]
                max_other_score = max(scores_j)
                threshold = 0.85
                
                if label_i == label_j:
                    if max_same_score > threshold:
                        true_positive = true_positive+1
                    else:
                        false_negative = false_negative+1
                else:
                    if max_other_score > threshold:
                        false_positive = false_positive+1
                    else:
                        true_negative = true_negative+1
                
                results_by_class[label_j].append([max_same_score, is_same, label_i, label_j])
                
        return results, results_by_class
    
    def _close_set(self, data, y, grid_clf, groups=None):
        average_session_results=[]
        cv=LeaveOneGroupOut()
        for train, test in cv.split(data, y, groups):
            X_train, X_test, y_train, y_test = data[train], data[test], y[train], y[test]
            tf.keras.backend.clear_session()
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(128)
            model=grid_clf._siamese_embeddings(X_train.shape[1], X_train.shape[2])
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
            history = model.fit(train_dataset,
                                        workers=5,
                                        callbacks=[early_stopping_callback])
            results=self._predict(model, X_test, y_test) 
            average_session_results.append(results) 
         
        return average_session_results
    
    def _open_set(self, X, y, grid_clf, groups=None):
        average_session_results=[]
        cv = LeaveOneGroupOut()
        for train_index, test_index in cv.split(X, y, groups=groups):
            X_train, X_test = [train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Find the unique subject IDs in the training set
            train_subjects = np.unique(y_train)

            # Exclude the subjects used in training from the test set
            test_subjects = np.setdiff1d(np.unique(y_test), train_subjects)

            # Filter the test set to include only the test subjects
            test_mask = np.isin(y_test, test_subjects)
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]
            tf.keras.backend.clear_session()
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(128)

            model=grid_clf._siamese_embeddings(X_train.shape[1], X_train.shape[2])
            early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
            history = model.fit(train_dataset,
                                        workers=5,
                                        callbacks=[early_stopping_callback])
            results=self._predict(model, X_test, y_test) 
            average_session_results.append(results) 
         
        return average_session_results
        
        #return average_session_results 
    def _evaluate(self, dataset, pipelines):
        if not self.is_valid(dataset):
            raise AssertionError("Dataset is not appropriate for evaluation")

        y=[]
        X, _, metadata=self.paradigm.get_data(dataset)
        if(dataset.paradigm == "p300"):
            target_index=metadata[metadata['event_id']=="Target"].index.tolist()
            metadata=metadata[metadata['event_id']=="Target"]

        elif (dataset.paradigm == "n400"):
            target_index=metadata[metadata['event_id']=="Inconsistent"].index.tolist()
            metadata=metadata[metadata['event_id']=="Inconsistent"]
        
        # Selecting the target trials if paradigm is p300 or inconsistent if paradigm is n400
        data=X[target_index]

        # Selecting the subject labels for the target or inconsistent trails
        #y=metadata.iloc[target_index]["subject"].tolist()
        y=np.array(metadata["subject"])

        # iterate over sessions
        # for session in np.unique(metadata.session):
        #     ix = metadata.session == session

        groups = metadata.session.values
        for name, clf in pipelines.items():
            
            grid_clf = clone(clf)
            if self.return_close_set:
                close_set_scores=self._close_set(data, y, grid_clf, groups=groups)

            elif self.return_open_set:
                open_set_scores=self._open_set(data, y, grid_clf, groups=groups)          
        return 0
    
    def is_valid(self, dataset):
        return dataset.n_sessions > 1

# class Siamese_OpenSetEvaluation(BaseEvaluation):
#     def __init__(
#         self,
#         n_perms: Optional[Union[int, Vector]] = None,
#         data_size: Optional[dict] = None,
#         # dataset=None,
#         # paradigm=None,
#         #paradigm=None,
#         **kwargs
#     ):
#         # self.dataset = dataset
#         # self.paradigm = paradigm
#         #self.paradigm = paradigm
#         self.n_perms = n_perms
#         self.data_size = data_size
#         super().__init__(**kwargs)

#     def _authenticate(self, embedding_network, x_test, y_test):
#         # Compute embeddings for all test samples

#         results = []
#         results_by_class = defaultdict(list)
#         labels = np.unique(y_test)
#         true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0
#         digit_indices = [np.where(y_test == i)[0] for i in labels]
        
#         # Compute embeddings for all test samples
#         test_embeddings = []
#         for c in range(0, len(x_test), 500):
#             embeddings_chunk = embedding_network(x_test[c:c+500])
#             test_embeddings.append(embeddings_chunk)
#         test_embeddings = tf.concat(test_embeddings, axis=0)
        
#         # Compute cosine similarity between all pairs of embeddings
#         pairwise_similarities = cs(test_embeddings, test_embeddings)
        
#         for i in range(len(x_test)):
#             # Compute scores for all other samples
#             scores = pairwise_similarities[i]
#             same_in = digit_indices[np.where(labels == y_test[i])[0][0]]
            
#             for j in range(len(x_test)):
#                 score = scores[j]
#                 label_i, label_j = y_test[i], y_test[j]
#                 is_same = int(j in same_in)
#                 results.append([score, is_same, label_i, label_j])
                
#                 # Update results by class
#                 results_by_class[label_j].append([score, is_same, label_i, label_j])
                
#             # Compute scores for same-class samples
#             same_scores = [pairwise_similarities[i][j] for j in same_in if j != i]
#             max_same_score = max(same_scores)
            
#             for label_j in labels:
#                 is_same = int(label_j == label_i)
#                 scores_j = [pairwise_similarities[j][k] for j in digit_indices[np.where(labels == label_j)[0][0]] for k in digit_indices[np.where(labels == label_j)[0][0]] if j != i]
#                 max_other_score = max(scores_j)
#                 threshold = 0.85
                
#                 if label_i == label_j:
#                     if max_same_score > threshold:
#                         true_positive = true_positive+1
#                     else:
#                         false_negative = false_negative+1
#                 else:
#                     if max_other_score > threshold:
#                         false_positive = false_positive+1
#                     else:
#                         true_negative = true_negative+1
                
#                 results_by_class[label_j].append([max_same_score, is_same, label_i, label_j])
                
#         return results, results_by_class


#     def _evaluate(self, dataset, pipelines):
#         #X=[]
#         y=[]
#         X, _,metadata=self.paradigm.get_data(dataset)
#         if(dataset.paradigm == "p300"):
#             data=[epochs['Target'] for epochs in X]
#         elif (dataset.paradigm == "n400"):
#             data=[epochs['Inconsistent'] for epochs in X]    
#         y=metadata['subject'].tolist()

#         # Defining the Stratified KFold

        # for name, clf in pipelines.items():
        #     groupfold = GroupKFold(n_splits=4,shuffle=True,random_state=42)

        #     for train_index, test_index in groupfold.split(data, y, groups=y):
        #         X_train, X_test = X[train_index], X[test_index]
        #         y_train, y_test = y[train_index], y[test_index]
        #         tf.keras.backend.clear_session()
        #         grid_clf = clone(clf)
        #         train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(1000).batch(128)
        #         model=grid_clf.fit(train_dataset, self.epoch, self.verbose) 
        #         results, results_by_class=self._authenticate(model, X_test, y_test) 
        # return 0

    