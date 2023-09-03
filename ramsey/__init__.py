"""
ramsey: probabilistic modelling using JAX
"""


from ramsey._src.attention.attention import Attention
from ramsey._src.attention.multihead_attention import MultiHeadAttention
from ramsey._src.neural_process.attentive_neural_process import ANP
from ramsey._src.neural_process.doubly_attentive_neural_process import DANP
from ramsey._src.neural_process.neural_process import NP
from ramsey._src.neural_process.train_neural_process import train_neural_process
from ramsey._src.nn.MLP import MLP

__version__ = "0.2.0"

__all__ = [
    "Attention",
    "ANP",
    "DANP",
    "MLP",
    "MultiHeadAttention",
    "NP",
    "train_neural_process",
]
