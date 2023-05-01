"""
An evaluation defines how we go from trials per subject and session to a
generalization statistic (AUC score, f-score, accuracy, etc) -- it can be
either within-recording-session accuracy, across-session within-subject
accuracy, across-subject accuracy, or other transfer learning settings.
"""
# flake8: noqa
from .evaluation_old import (
    CloseSetEvaluation,
    OpenSetEvaluation,
)

from .cross_session_evaluation import CrossSessionEvaluation
from .within_session_evaluation import WithinSessionEvaluation
from .siamese_evaluation import (Siamese_WithinSessionEvaluation, Siamese_CrossSessionEvaluation)