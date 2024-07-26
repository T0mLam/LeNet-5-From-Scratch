from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from nptyping import NDArray, Number


class Loss(ABC):
    @abstractmethod
    def forward(self, y: NDArray, y_pred: NDArray) -> Number:
        pass

    @abstractmethod
    def backward(self) -> NDArray:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
                       

class MSE(Loss):
    def forward(self, 
        y: NDArray[Any, Number],
        y_pred: NDArray[Any, Number]
    ) -> float:
        self.y = y
        self.y_pred = y_pred
        return np.mean((self.y_pred - self.y) ** 2)
    
    def backward(self) -> NDArray[Any, Number]: 
        return 2 * (self.y_pred - self.y) / self.y.shape[0]
    

class CrossEntropy(Loss):
    def forward(self, 
        y: NDArray[Any, Number],
        y_pred: NDArray[Any, Number]
    ) -> float:
        self.y = y
        self.y_pred = y_pred.clip(min=1e-8)
        self.y_one_hot = np.eye(self.y_pred.shape[1])[y]
        return -np.sum(self.y_one_hot * np.log(self.y_pred)) / self.y.shape[0]

    def backward(self) -> NDArray[Any, Number]:
        return (self.y_pred - self.y_one_hot) / self.y.shape[0]
    

class BinaryEntropy(Loss):
    pass