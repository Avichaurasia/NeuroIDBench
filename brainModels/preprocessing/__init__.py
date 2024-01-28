"""
A paradigm defines how the raw data will be converted to trials ready to be
processed by a decoding algorithm. This is a function of the paradigm used
"""

# from deeb.paradigms.p300 import P300
# from deeb.paradigms.n400 import N400
from .base import BaseParadigm
print("base imported")
from .erp import BaseERP, SinglePass
print("erp imported from paradigms")
from .erp import ERP
#from deeb.paradigms.n400 import N400
