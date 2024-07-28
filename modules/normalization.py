from __future__ import annotations
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

        Momentum Batch Normalization for Deep Learning with Small Batch Size
        https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123570222.pdf
    """

    def __init__(
            self, 
            channels: int,
            epsilon: float=1e-5,
            momentum: float=0.1
        ) -> None:
        self.gamma = np.ones((1, channels, 1, 1))
        self.beta = np.zeros((1, channels, 1, 1))
        self.epsilon = epsilon

        self.running_mean = np.zeros((1, channels, 1, 1))
        self.running_variance = np.zeros((1, channels, 1, 1))
        self.momentum = momentum
        
    def forward(
        self, 
        X: NDArray[Shape['*, *, *, *'], Number],
        train: bool=True
    ) -> NDArray[Shape['*, *, *, *'], Number]:
        self.X = X
        self.m = X.shape[0]

        if train:
            self.train_forward()
        else:
            self.test_forward()
        
        return self.Y

    def train_forward(self) -> None:
        self.mu = np.mean(self.X, axis=(0, 2, 3), keepdims=True)
        self.sigma2 = np.var(self.X, axis=(0, 2, 3), keepdims=True)
         
        self.X_hat = (self.X - self.mu) / np.sqrt(self.sigma2 + self.epsilon)
        self.Y = self.gamma * self.X_hat + self.beta

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mu
        self.running_variance = self.momentum * self.running_variance + (1 - self.momentum) * self.sigma2

    def test_forward(self) -> None:      
        self.X_hat = (self.X - self.running_mean) / np.sqrt(self.running_variance + self.epsilon)
        self.Y = self.gamma * self.X_hat + self.beta
        
    def backward(
        self,
        grad: NDArray[Shape['*, *, *, *'], Number]
    ) -> NDArray[Shape['*, *, *, *'], Number]:
        self.dX_hat = grad * self.gamma
        self.dsigma2 = np.sum(
            self.dX_hat * (self.X - self.mu) * -0.5 * (self.sigma2 + self.epsilon) ** -1.5,
            keepdims=True
        )
        block1 = np.sum(
            self.dX_hat / -np.sqrt(self.sigma2 + self.epsilon), 
            axis=(0, 2, 3),
            keepdims=True
        )
        block2 = self.dsigma2 * np.sum(-2 * (self.X - self.mu), keepdims=True) / self.m
        self.dmu = block1 + block2
        
        block1 = self.dX_hat / np.sqrt(self.sigma2 + self.epsilon) 
        block2 = self.dsigma2 * 2 * (self.X - self.mu) / self.m
        block3 = self.dmu / self.m
        self.dX = block1 + block2 + block3

        self.dgamma = np.sum(grad * self.X_hat, axis=(0, 2, 3), keepdims=True)
        self.dbeta = np.sum(grad, axis=(0, 2, 3), keepdims=True)

        return self.dX