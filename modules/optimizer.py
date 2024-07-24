from abc import ABC, abstractmethod

from .layer import Linear, Conv

from .sequential import Sequential


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
                block.K -= s