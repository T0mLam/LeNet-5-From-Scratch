from __future__ import annotations
from abc import ABC, abstractmethod

from .layer import Linear, Conv


class Optimizer(ABC):
    def __init__(self, model: Sequential, lr: float) -> None:
        self.model = model
        self.lr = lr

    @abstractmethod
    def step(self):
        pass


class GradDesc(Optimizer):
    def step(self):
        for block in self.model.blocks:
            if isinstance(block, Linear):
                block.W -= self.lr * block.dW
                block.b -= self.lr * block.db

            if isinstance(block, Conv):
                block.K -= self.lr * block.dK
                block.b -= self.lr * block.db


class Adam(Optimizer):
    pass