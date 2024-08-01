import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms 

from modules.activation import ReLU, SoftMax, Tanh, SquashedTanh
from modules.criterion import MSE, CrossEntropy
from modules.init import Xavier
from modules.layer import Linear, Conv, Flatten
from modules.model import Sequential, train, test
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

init_method = Xavier()

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
    Linear(84, 10, init_method), #output
    SoftMax()
])

model.train()

criterion = CrossEntropy()
optimizer = Adam(model, lr=0.01)

train_acc, train_loss = train(
    model, X_train, y_train, criterion, optimizer, 3, 128
)

plt.plot(train_acc)
plt.show()

"""
acc = 0
for i in range(len(X_test)):
    pred = model(X_test[i].reshape(1, 784)).argmax()
    if pred == y_test[i]:
        acc += 1

print(f'Test accuracy: {acc / len(X_test) * 100}%')

test_acc = test(
    model, X_test, y_test, 200
)
print(test_acc)
"""