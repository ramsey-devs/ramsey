"""
ramsey: probabilistic modelling with Haiku and JAX
"""

from ramsey._src.gaussian_process.gaussian_process import GP
from ramsey._src.gaussian_process.sparse_gaussian_process import SparseGP
from ramsey._src.neural_process.attentive_neural_process import ANP
from ramsey._src.neural_process.doubly_attentive_neural_process import DANP
from ramsey._src.neural_process.neural_process import NP

__version__ = "0.1.0"

__all__ = ["NP", "ANP", "DANP", "GP", "SparseGP"]
