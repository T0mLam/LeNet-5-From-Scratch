from typing import Any, List

import numpy as np
from nptyping import Number, NDArray, Shape
from tqdm import tqdm

from .layer import Layer


class Sequential:
    def __init__(self, blocks: List[Layer]) -> None:
        self.blocks = blocks

    def forward(
        self, 
        X: NDArray[Any, Number]
    ) -> NDArray[Any, Number]:
        for block in self.blocks:
            X = block(X)
        return X

    def backward(
        self, 
        grad: NDArray[Any, Number]
    ) -> None:
        for block in reversed(self.blocks):
            grad = block.backward(grad)
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

def train(
    model, 
    X_train, 
    y_train, 
    criterion,
    optimizer,
    epochs,
    batch_size
):
    loss_list = []
    acc_list = []
    iterations = len(X_train) // batch_size

    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}')
        correct = 0
        loss = 0

        for i in tqdm(range(iterations), desc='Training'):
            lbound, ubound = i * batch_size, (i + 1) * batch_size
            X = X_train[lbound: ubound]
            y = y_train[lbound: ubound]
            #y_pred = model(X.reshape(batch_size, -1))
            y_pred = model(X)
            correct += np.sum(np.argmax(y_pred, axis=1) == y)
            loss += criterion(y, y_pred)
            grad = criterion.backward()
            model.backward(grad)
            optimizer.step()

        acc = correct / len(y_train)
        acc_list.append(acc)
        loss_list.append(loss)
        print(f'Accuracy: {acc} | Loss: {loss}')

    return acc_list, loss_list


def test(
    model,
    X_test,
    y_test,
    batch_size
):
    correct = 0
    iterations = len(X_test) // batch_size

    for i in tqdm(range(iterations), desc='Testing'):
        lbound, ubound = i * batch_size, (i + 1) * batch_size
        X = X_test[lbound: ubound]
        y = y_test[lbound: ubound]
        y_pred = np.argmax(model(X.reshape(batch_size, -1)), axis=1)
        correct += np.sum(y_pred == y)
        
    return correct / len(y_test)