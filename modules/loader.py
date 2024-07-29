from typing import Tuple

from nptyping import NDArray, Shape, Number


class DatasetLoader:
    def __init__(
        self, 
        X: NDArray[Shape['*, *, ...'], Number],
        y: NDArray[Shape['*, ...'], Number],
        batch_size: int=1,
        shuffle: bool=False
    ) -> None:
        self.X = X
        self.y = y
        self.n_samples = len(X)
        self.iter = self.n_samples // batch_size
        self.batch_size = batch_size

    def __getitem__(
            self, 
            idx: int
    ) -> Tuple[
        NDArray[Shape['*, *, ...'], Number], 
        NDArray[Shape['*, ...'], Number]
    ]:
        return self.X[idx], self.y[idx]
    
    def __iter__(self) -> NDArray[Shape['*, ...'], Number]:
        for i in range(self.iter):
            lbound, ubound = i * self.batch_size, (i + 1) * self.batch_size
            X_subset = self.X[lbound: ubound]
            y_subset = self.y[lbound: ubound]
            yield X_subset, y_subset
    
    def __len__(self) -> int:
        return self.iter