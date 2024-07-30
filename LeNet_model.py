import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms 

from modules.activation import ReLU, SoftMax, Tanh
from modules.criterion import MSE, CrossEntropy
from modules.init import Xavier
from modules.layer import Linear, Conv, Flatten
from modules.model import Sequential, train, test
from modules.optimizer import GradDesc, Adam
from modules.regularization import Dropout
from modules.normalization import BatchNorm
from modules.pooling import AvgPool


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
    Conv((32, 32), 128, 1, 6, 5), #c1
    AvgPool(2), #s2
    Tanh(),
    Conv((14, 14), 128, 6, 16, 5), #c3
    AvgPool(2), #s4
    Tanh(), 
    Conv((5, 5), 128, 16, 120, 5), #c5
    Tanh(),
    Flatten(),
    Linear(120, 84, init_method), #f6
    Tanh(),
    Linear(84, 10, 