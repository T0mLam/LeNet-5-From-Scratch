from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, List, Tuple    

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from torchvision import datasets, transforms 
from nptyping import Number, NDArray, Shape

from modules.activation import ReLU, SoftMax, Tanh, SquashedTanh
from modules.criterion import MSE, CrossEntropy
from modules.init import Xavier, LeCun
from modules.layer import Linear, Conv, Flatten, RBF
from modules.loader import DatasetLoader
from modules.model import train, test
from modules.optimizer import GradDesc, Adam
from modules.regularization import Dropout
from modules.normalization import BatchNorm
from modules.pooling import AvgPool
from constant import ASCII_BITMAP, C3_MAPPING


composed_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(0, 1)
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=composed_transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=composed_transform)

X_train = np.array([data[0].numpy() for data in train_dataset])[:2560]
y_train = np.array([data[1] for data in train_dataset])[:2560]
X_test = np.array([data[0].numpy() for data in test_dataset])[:2560]
y_test = np.array([data[1] for data in test_dataset])[:2560]

init_method = LeCun()

class Sequential:
    def __init__(self, blocks: List[Layer]) -> None:
        self.blocks = blocks
        self.is_training = True    

    def forward(
        self, 
        X: NDArray[Any, Number],
        **kwargs
    ) -> NDArray[Any, Number]:
        for block in self.blocks:
            if self.is_training:
                X = block(X, train=self.is_training, y=kwargs['y'])
            else:
                X = block(X, train=self.is_training)
        return X

    def backward(
        self, 
        grad: NDArray[Any, Number]
    ) -> None:
        for block in reversed(self.blocks):
            grad = block.backward(grad)

    def train(self) -> None:
        self.is_training = True

    def eval(self) -> None:
        self.is_training = False
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

model = Sequential([
    # Conv Layers
    Conv((32, 32), 128, 1, 6, 5), #c1
    AvgPool(2), #s2
    SquashedTanh(),
    Conv((14, 14), 128, 6, 16, 5, mapping=C3_MAPPING), #c3
    AvgPool(2), #s4
    SquashedTanh(), 
    Conv((5, 5), 128, 16, 120, 5), #c5
    SquashedTanh(),
    
    # FC Layers
    Flatten(),
    Linear(120, 84, init_method), #f6
    SquashedTanh(),
    RBF(ASCII_BITMAP), #output
])

model.train()

criterion = MSE()
optimizer = Adam(model, lr=0.01)
batch_size = 128
epochs = 3

loss_list = []
acc_list = []
train_loader = DatasetLoader(X_train, y_train, batch_size=batch_size)
test_loader = DatasetLoader(X_test, y_test, batch_size=batch_size)

for epoch in range(epochs):
    print(f'\nEpoch {epoch + 1}')
    loss = 0
    model.train()

    for X, y in tqdm(train_loader, desc='Training'):
        y_pred = model(X, y=y)
        loss += criterion(y, y_pred)
        grad = criterion.backward()
        model.backward(grad)
        optimizer.step()

    correct = 0
    model.eval()

    for X, y in tqdm(test_loader, desc='Testing'):
        y_pred = model(X)
        correct += np.sum(y_pred == y)

    acc = correct / len(y_test)
    acc_list.append(acc)
    loss_list.append(loss)
    print(f'Accuracy: {acc} | Loss: {loss}')

plt.plot(loss_list)
plt.show()

"""
acc = 0
for i in range(len(X_test)):
    pred = model(X_test[i]).argmax()
    if pred == y_test[i]:
        acc += 1

print(f'Test accuracy: {acc / len(X_test) * 100}%')
"""