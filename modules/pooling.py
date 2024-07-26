from typing import Any, Optional
from abc import ABC, abstractmethod

import numpy as np
from nptyping import NDArray, Number, Shape

from .layer import Layer


class Pool(Layer):
    def forward(
        self, 
        X: NDArray[Shape['*, *, *, *'], Number],
        kernel_size: int,
        stride: Optional[int]=None,
        train: bool=True
    ) -> NDArray[Shape['*, *, *, *'], Number]:
        if stride is None: 
            stride = kernel_size

        self.X = X
        self.kernel_size = kernel_size
        self.stride = stride
        self.batch_size, self.channels, self.height, self.width = X.shape

        self.output_height = (self.height - kernel_size) // stride + 1
        self.output_width = (self.width - kernel_size) // stride + 1

        self.Y = np.zeros((
            self.batch_size, self.channels, self.output_height, self.output_width
        ))

        for b in range(self.batch_size):
            for ch in range(self.channels):
                for r in range(self.output_height):
                    for c in range(self.output_width): 
                        r_start = r * self.stride
                        c_start = c * self.stride
                        r_end = r_start + self.kernel_size
                        c_end = c_start + self.kernel_size
                        self.pool(b, ch, r, c, r_start, c_start, r_end, c_end)

        return self.Y
    
    @abstractmethod
    def pool(
        self, 
        b: int, 
        ch: int, 
        r: int, 
        c: int, 
        r_start: int, 
        c_start: int, 
        r_end: int, 
        c_end: int
    ) -> None:
        pass

    def backward(
        self,
        grad: NDArray[Shape['*, *, *, *'], Number]
    ) -> NDArray[Shape['*, *, *, *'], Number]:
        self.grad = grad
        self.output = np.zeros(self.X.shape)

        for b in range(self.batch_size):
            for ch in range(self.channels):
                for r in range(self.output_height): 
                    for c in range(self.output_width): 
                            r_start = r * self.stride
                            c_start = c * self.stride
                            r_end = r_start + self.kernel_size
                            c_end = c_start + self.kernel_size
                            self.backward_pool(b, ch, r, c, r_start, c_start, r_end, c_end)

        return self.output

    @abstractmethod
    def backward_pool(
        self, 
        b: int, 
        ch: int, 
        r: int, 
        c: int, 
        r_start: int, 
        c_start: int, 
        r_end: int, 
        c_end: int
    ) -> None:
        pass


class MaxPool(Pool):
    def pool(
        self, 
        b: int, 
        ch: int, 
        r: int, 
        c: int, 
        r_start: int, 
        c_start: int, 
        r_end: int, 
        c_end: int
    ) -> None:
        self.Y[b, ch, r, c] = np.max(
            self.X[b, ch, r_start: r_end, c_start: c_end]
        )

    def backward_pool(
        self, 
        b: int, 
        ch: int, 
        r: int, 
        c: int, 
        r_start: int, 
        c_start: int, 
        r_end: int, 
        c_end: int
    ) -> None:
        window = self.X[b, ch, r_start: r_end, c_start: c_end]
        max_val = self.Y[b, ch, r, c]

        indicies = np.where(window == max_val)

        i, j = indicies[0][0], indicies[1][0]
        self.output[b, ch, i + r_start, j + c_start] = self.grad[b, ch, r, c]


class AvgPool(Pool):
    def pool(
        self, 
        b: int, 
        ch: int, 
        r: int, 
        c: int, 
        r_start: int, 
        c_start: int, 
        r_end: int, 
        c_end: int
    ) -> None:
        self.Y[b, ch, r, c] = np.mean(
            self.X[b, ch, r_start: r_end, c_start: c_end]
        )

    def backward_pool(
        self, 
        b: int, 
        ch: int, 
        r: int, 
        c: int, 
        r_start: int, 
        c_start: int, 
        r_end: int, 
        c_end: int
    ) -> None:
        window = self.output[b, ch, r_start: r_end, c_start: c_end]
        avg_grad = self.grad[b, ch, r, c] / self.kernel_size ** 2

        window += avg_grad


if __name__ == '__main__':
    img = np.array([[[[3, 6, 4, 3], [9, 5, 1, 2], [1, 5, 1, 8], [1, 7, 2, 4]]]])
    mp = AvgPool()
    print(img)
    print(mp(img, 2, 1))
    print(mp.backward(np.array([[[[9, 6, 4], [9, 5, 8], [7, 7, 8]]]])))