import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data

from modules.activation import ReLU, SoftMax, Tanh
from modules.criterion import MSE, CrossEntropy
from modules.init import Kaiming
from modules.layer import Linear, Conv, Flatten
from modules.model import Sequential, train, test
from modules.optimizer import GradDesc
from modules.regularization import Dropout



(X_train, y_train), (X_test, y_test) = load_data()
X_train = (X_train - 127.5) / 127.5
X_test = (X_test - 127.5) / 127.5

init_method = Kaiming()

model = Sequential([
    Conv(1, 8, 4),
    ReLU(),
    Dropout(0.2),
    Conv(8, 8, 4),
    ReLU(),
    Flatten(),
    Linear(22 * 22 * 8, 64, init_method),
    ReLU(),
    Linear(64, 10, init_method),
    SoftMax()
])

criterion = CrossEntropy()
optimizer = GradDesc(model, lr=0.01)

train_acc, train_loss = train(
    model, X_train, y_train, criterion, optimizer, 3, 1600
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