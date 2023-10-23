"""
ramsey:  Probabilistic deep learning using JAX
"""

from ramsey._src.neural_process.attentive_neural_process import ANP
from ramsey._src.neural_process.doubly_attentive_neural_process import DANP
from ramsey._src.neural_process.neural_process import NP
from ramsey._src.neural_process.train_neural_process import train_neural_process

__version__ = "0.2.1"

__all__ = [
    "ANP",
    "DANP",
    "NP",
    "train_neural_process",
]
