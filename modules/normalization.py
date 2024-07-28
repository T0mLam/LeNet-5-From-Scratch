from typing import Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from nptyping import NDArray, Number, Shape

from .layer import Layer


class BatchNorm(Layer):
    """
    Reference:
        Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
        https://arxiv.org/pdf/1502.03167
    """
    def __init__(self) -> None:
        self.gamma = None
        self.beta = None
        self.epsilon = 1e-8

    def forward(
        self, 
        X: NDArray[Any, Number],
        train: bool=True
    ) -> NDArray:
        self.X = X
        self.m = np.prod(X.shape)

        if train:
            self.train_forward()
        else:
            self.test_forward()

    def train_forward(self)-> NDArray[Shape['*, *, *, *'], Number]:
        self.mu = np.mean(self.X)
        self.sigma2 = np.var(self.X)
         
        self.X_hat = (self.X - self.mu) / np.sqrt(self.sigma2 + self.epsilon)
        self.Y = self.gamma * self.X_hat + self.beta

        return self.Y

    def test_forward(self) -> NDArray[Shape['*, *, *, *'], Number]:      
        pass
        
    def backward(
        self,
        grad: NDArray[Shape['*, *, *, *'], Number]
    ) -> NDArray[Shape['*, *, *, *'], Number]:
        self.dX_hat = grad * self.gamma
        self.dsigma2 = np.sum(
            self.dX_hat * (self.X - self.mu) * -0.5 * (self.sigma2 + self.epsilon) ** (-1.5)
        )
        block1 = np.sum(self.dX_hat / -np.sqrt(self.sigma2 + self.epsilon))
        block2 = self.dsigma2 * np.sum(-2 * (self.X - self.mu)) / self.m
        self.dmu = block1 + block2
        
        block1 = self.dX_hat * np.sqrt(self.dsigma2 + self.epsilon) ** -1
        block2 = self.dsigma2 * 2 * (self.X - self.mu) / self.m
        block3 = self.dmu / self.m
        self.dX = block1 + block2 + block3

        self.dgamma = np.sum(grad * self.X_hat)
        self.dbeta = np.sum(grad)

        return self.dX