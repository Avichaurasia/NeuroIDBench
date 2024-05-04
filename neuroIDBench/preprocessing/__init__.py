"""
A paradigm defines how the raw data will be converted to trials ready to be
processed by a decoding algorithm. This is a function of the paradigm used
"""

from .base import BaseParadigm
from .erp import BaseERP, SinglePass
from .erp import ERP
