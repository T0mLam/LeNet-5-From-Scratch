import torch
from torch import nn 


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), #c1
            nn.Tanh(),
            nn.AvgPool2d(2), #s2
            nn.Tanh(),
            nn.Conv2d(6, 16, 5), #c3
            nn.Tanh(),
            nn.AvgPool2d(2), #s4
            nn.Tanh(),
            nn.Conv2d(16, 120, 5), #c5
            nn.Tanh(),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(120, 84), #f6
            nn.Tanh(),
            nn.Linear(84, 10) #output
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x
    

if __name__ == '__main__':
    # Load pre-trained model
    path = 'pytorch_model/trained_models/LeNet_5_Model_StateDict.pth'

    model = LeNet5()
    model.load_state_dict(torch.load(path))