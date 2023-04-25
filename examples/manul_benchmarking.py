import sys
sys.path.append('/Users/avinashkumarchaurasia/Master_Thesis/deeb/')

# Importing the datasets
from deeb.datasets.brainInvaders15a import BrainInvaders2015a
from deeb.datasets.mantegna2019 import Mantegna2019
from deeb.datasets.draschkow2018 import Draschkow2018
from deeb.datasets.won2022 import Won2022

# Importing the paradigms
from deeb.paradigms.n400 import N400
from deeb.paradigms.p300 import P300

# Importing the pipelines
from deeb.pipelines.features import AutoRegressive 
from deeb.pipelines.features import PowerSpectralDensity 

# Importing the evaluation
#from deeb.evaluation.evaluation import CloseSetEvaluation
#from deeb.evaluation.evaluation import OpenSetEvaluation

# Importing the classifiers
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline


# Intiaizing the datasets
won = Won2022()
brain=BrainInvaders2015a()
mantegna=Mantegna2019()


# Intializing the paradigms
paradigm=P300()
paradigm_n400=N400()


# Intializing the pipelines
pipeline={}
pipeline['AR+SVM']=make_pipeline(AutoRegressive(order=10), SVC(kernel='rbf', probability=True))
pipeline['PSD+SVM']=make_pipeline(PowerSpectralDensity(), SVC(kernel='rbf', probability=True))


# Intializing the evaluation for closed set and open set
close_set=CloseSetEvaluation(paradigm=paradigm_n400, datasets=mantegna, overwrite=False)
open_set=OpenSetEvaluation(paradigm=paradigm_n400, datasets=mantegna, overwrite=False)


# getting the scores for closed set and open set in the form of dataframe
results_close_set=close_set.process(pipeline)
results_open_set=open_set.process(pipeline)









