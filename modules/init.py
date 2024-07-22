from abc import ABC, abstractmethod

import numpy as np


class Initialization(ABC):
    @abstractmethod
    def __call__(self, in_dim, out_dim):
        pass


class Kaiming(Initialization):
    def __call__(self, in_dim, out_dim):
        var = 2 / in_dim
        W = np.random.randn(out_dim, in_dim) * np.sqrt(var)
        b = np.zeros(out_dim)
        return W, b