from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np

from .layer import Linear, Conv
from .normalization import BatchNorm


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

            elif isinstance(block, Conv):
                block.K -= self.lr * block.dK
                block.b -= self.lr * block.db

            elif isinstance(block, BatchNorm):
                block.gamma -= self.lr * block.dgamma
                block.beta -= self.lr * block.dbeta


class Adam(Optimizer):
    """
    Reference:
        Adam: A Method For Stochastic Optimization
        https://arxiv.org/pdf/1412.6980
    """
    def __init__(
        self, 
        model: Sequential, 
        lr: float=0.001,
        beta1: float=0.9,
        beta2: float=0.999,
        epsilon: float=1e-8
    ) -> None:
        super().__init__(model, lr)
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}

        for block in self.model.blocks:
            if isinstance(block, Linear):
                W_id, b_id = id(block.W), id(block.b)
                self.m[W_id] = np.zeros_like(block.W)
                self.v[W_id] = np.zeros_like(block.W)
                self.m[b_id] = np.zeros_like(block.b)
                self.v[b_id] = np.zeros_like(block.b)

            elif isinstance(block, Conv):
                K_id, b_id = id(block.K), id(block.b)
                self.m[K_id] = np.zeros_like(block.K)
                self.v[K_id] = np.zeros_like(block.K)
                self.m[b_id] = np.zeros_like(block.b)
                self.v[b_id] = np.zeros_like(block.b)

            elif isinstance(block, BatchNorm):
                gamma_id, beta_id = id(block.gamma), id(block.beta)
                self.m[gamma_id] = np.zeros_like(block.gamma)
                self.v[gamma_id] = np.zeros_like(block.gamma)
                self.m[beta_id] = np.zeros_like(block.beta)
                self.v[beta_id] = np.zeros_like(block.beta)

    def step(self):
        self.t += 1
        lr_t = self.lr * np.sqrt(1 - self.beta2 ** self.t) / (1 - self.beta1 ** self.t)

        for block in self.model.blocks:
            if isinstance(block, Linear):
                block.W -= lr_t * self.update_rate(id(block.W), block.dW)
                block.b -= lr_t * self.update_rate(id(block.b), block.db)

            elif isinstance(block, Conv):
                block.K -= lr_t * self.update_rate(id(block.K), block.dK)
                block.b -= lr_t * self.update_rate(id(block.b), block.db)

            elif isinstance(block, BatchNorm):
                block.gamma -= lr_t * self.update_rate(id(block.gamma), block.dgamma)
                block.beta -= lr_t * self.update_rate(id(block.beta), block.dbeta)

    def update_rate(self, param_id, grad):
        m = self.m[param_id]
        v = self.v[param_id]

        m_t = self.beta1 * m + (1 - self.beta1) * grad
        v_t = self.beta2 * v + (1 - self.beta2) * grad ** 2

        self.m[param_id] = m_t
        self.v[param_id] = v_t

        m_hat = m_t / (1 - self.beta1 ** self.t)
        v_hat = v_t / (1 - self.beta2 ** self.t)

        return m_hat / (np.sqrt(v_hat) + self.epsilon)