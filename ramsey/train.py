from ramsey._src.gaussian_process.train_gaussian_process import (
    train_gaussian_process,
    train_sparse_gaussian_process,
)
from ramsey._src.neural_process.train_neural_process import train_neural_process

__all__ = [
    "train_neural_process",
    "train_gaussian_process",
    "train_sparse_gaussian_process",
]
