import matplotlib.pyplot as plt
from keras.datasets.mnist import load_data

from modules.layer import Linear
from modules.activation import ReLU, SoftMax, Tanh
from modules.criterion import MSE, CrossEntropy
from modules.optimizer import GradDesc
from modules.sequential import Sequential, train, test
from modules.init import Kaiming


(X_train, y_train), (X_test, y_test) = load_data()
X_train = (X_train - 127.5) / 127.5
X_test = (X_test - 127.5) / 127.5

init_method = Kaiming()

model = Sequential([
    Linear(28 * 28, 64, init_method),
    ReLU(),
    Linear(64, 32, init_method),
    ReLU(),
    Linear(32, 10, init_method),
    SoftMax()
])

criterion = CrossEntropy()
optimizer = GradDesc(model, lr=0.01)

train_acc, train_loss = train(
    model, X_train, y_train, criterion, optimizer, 50, 200
)

plt.plot(train_acc)
plt.show()

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