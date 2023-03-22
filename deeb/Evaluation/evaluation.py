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
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
import pandas as pd
from sklearn.model_selection._validation import _fit_and_score, _score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler as sc
from deeb.Evaluation.base import BaseEvaluation
from imblearn.over_sampling import RandomOverSampler as oversampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score
import random
log = logging.getLogger(__name__)

# Numpy ArrayLike is only available starting from Numpy 1.20 and Python 3.8
Vector = Union[list, tuple, np.ndarray]


class CloseSetEvaluation(BaseEvaluation):
    VALID_POLICIES = ["per_class", "ratio"]

    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        **kwargs,
    ):
        self.data_size = data_size
        self.n_perms = n_perms
        self.calculate_learning_curve = self.data_size is not None
        super().__init__(**kwargs)


    def _grid_search(self, param_grid, name_grid, name, grid_clf, X_, y_, cv):
        # Load result if the folder exists
        if param_grid is not None and not os.path.isdir(name_grid):
            if name in param_grid:
                search = GridSearchCV(
                    grid_clf,
                    param_grid[name],
                    refit=True,
                    cv=cv,
                    n_jobs=self.n_jobs,
                    scoring=self.paradigm.scoring,
                    return_train_score=True,
                )
                search.fit(X_, y_)
                grid_clf.set_params(**search.best_params_)

                # Save the result
                os.makedirs(name_grid, exist_ok=True)
                joblib.dump(
                    search,
                    os.path.join(name_grid, "Grid_Search_WithinSession.pkl"),
                )
                del search
                return grid_clf

            else:
                return grid_clf

        elif param_grid is not None and os.path.isdir(name_grid):
            search = joblib.load(os.path.join(name_grid, "Grid_Search_WithinSession.pkl"))
            grid_clf.set_params(**search.best_params_)
            return grid_clf

        elif param_grid is None:
            return grid_clf
        
    def _authenticate_single_subject(self, X,y, pipeline, param_grid=None):

        # Defining the Stratified KFold
        skfold = StratifiedKFold(n_splits=4,shuffle=True,random_state=42)
        for name, clf in pipeline.items():
            clf=clf[1:]

            # Splitting the dataset into the Training set and Test set
            for fold, (train_index, test_index) in enumerate(skfold.split(X, y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Normalizing the data using StandardScaler
                X_train=sc.fit_transform(X_train)
                X_test=sc.transform(X_test)

                # Resampling the data using RandomOverSampler
                #oversampler = RandomOverSampler()
                X_train, y_train = oversampler.fit_resample(X_train, y_train)

                # Training the model
                model=clf.fit(X_train,y_train)

                # Predicting the test set result
                y_pred=model.predict(X_test)
                print(f"Accuracy: {accuracy_score(y_test,y_pred)}")
        return y_pred
         
    def _evaluate(self, dataset, pipelines, param_grid):
        for key, features in pipelines.items():
            data=features[0].get_data(dataset, self.paradigm)
            for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-CloseSetEvaulation"):
                dataset=data.copy(deep=True)
                dataset['Label']=0
                dataset.loc[dataset['Subject'] == subject, 'Label'] = 1
                labels=np.array(dataset['Label'])
                X=np.array(dataset.drop(['Label','Event_id','Subject'],axis=1))
                predictions=self._authenticate_single_subject(X,labels, pipelines[key], param_grid)

        return predictions
                

    def evaluate(self, dataset, pipelines, param_grid):
        if self.calculate_learning_curve:
            yield from self._evaluate_learning_curve(dataset, pipelines)
        else:
            yield from self._evaluate(dataset, pipelines, param_grid)

    def is_valid(self, dataset):
        return True
    
class OpenSetEvaluation(BaseEvaluation):
    VALID_POLICIES = ["per_class", "ratio"]

    def __init__(
        self,
        n_perms: Optional[Union[int, Vector]] = None,
        data_size: Optional[dict] = None,
        **kwargs,
    ):
        self.data_size = data_size
        self.n_perms = n_perms
        self.calculate_learning_curve = self.data_size is not None
        super().__init__(**kwargs)

    def _build_model(self, train_set, test_set, pipeline, param_grid=None):
        X_train=train_set.drop(['Subject', 'Event_id',"Event_id_PSD",'Label'], axis=1)
        X_train=np.array(X_train)
        y_train=np.array(train_set['Label'])

        # Diving training data into X_test and y_test
        X_test=test_set.drop(['Subject', 'Event_id',"Event_id_PSD",'Label'], axis=1)
        X_test=np.array(X_test)
        y_test=np.array(test_set['Label'])

         # Normalizing the data using StandardScaler
        X_train=sc.fit_transform(X_train)
        X_test=sc.transform(X_test)

        # Resampling the data using RandomOverSampler
        #oversampler = RandomOverSampler()
        X_train, y_train = oversampler.fit_resample(X_train, y_train)

        model=pipeline.fit(X_train, y_train)

        # Training the model
        model=pipeline.fit(X_train,y_train)

        # Predicting the test set result
        y_pred=model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test,y_pred)}")
        return y_pred



    def _authenticate_single_subject(self, dataset, df_authenticated, df_rejected, subject_ids, pipeline, param_grid=None, k=4):
        for fold in range(k):

            #Assigining 75% subjects subject_ids for training data
            train_subject_ids = random.sample(subject_ids, k=int(len(subject_ids) * 0.75))

            #Assigining 25% subjects subject_ids for training data
            test_subject_ids=dataset[~dataset['Subject'].isin(train_subject_ids)]['Subject'].unique()
            test_subject_ids=list(test_subject_ids)
            
            # Divide the dataset into training and testing sets based on subject id
            train_set = df_rejected[df_rejected['Subject'].isin(train_subject_ids)]
            test_set = df_rejected[df_rejected['Subject'].isin(test_subject_ids)]
            
            # Adding Authenticated subjects data in the training as well testing
            num_rows = int(len(df_authenticated) * 0.6)
            df_authenticated_train=df_authenticated.sample(n=num_rows)
            
            df_authenticated_test=df_authenticated.drop(df_authenticated_train.index)
            
            train_set=pd.concat([df_authenticated_train, train_set], axis=0)
            test_set=pd.concat([df_authenticated_test, test_set], axis=0)
            k=self._build_model(train_set, test_set, pipeline, param_grid)

    
        return k
         
    def _evaluate(self, dataset, pipelines, param_grid):
        for key, features in pipelines.items():
            data=features[0].get_data(dataset, self.paradigm)
            for subject in tqdm(dataset.subject_list, desc=f"{dataset.code}-CloseSetEvaulation"):
                dataset=data.copy(deep=True)
                dataset['Label']=0
                dataset.loc[dataset['Subject'] == subject, 'Label'] = 1
                df_authenticated=dataset[dataset['Subject']==subject]
                df_rejected=dataset.drop(df_authenticated.index)
                subject_ids = list(set(df_rejected['Subject']))
                #k=4
                c=self._authenticate_single_subject(dataset, df_authenticated, df_rejected, subject_ids, pipelines[key], param_grid, k=4)

        return c
                

    def evaluate(self, dataset, pipelines, param_grid):
        if self.calculate_learning_curve:
            yield from self._evaluate_learning_curve(dataset, pipelines)
        else:
            yield from self._evaluate(dataset, pipelines, param_grid)

    def is_valid(self, dataset):
        return True
    