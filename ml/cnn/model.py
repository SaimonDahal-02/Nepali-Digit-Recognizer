from torch import nn
import torch

class ConvNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvNeuralNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3))
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3))
        self.act2 = nn.GELU()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))

        self.flat = nn.Flatten()
        
        self.fc = nn.Linear(6272, 512)
        self.act3 = nn.GELU()
        self.drop2 = nn.Dropout(0.3)

        self.fc1 = nn.Linear(512, 64)
        self.act4 = nn.GELU()
        self.drop3 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 10)
        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.xavier_normal_(self.fc.weight)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.drop1(x)

        x = self.act2(self.conv2(x))
        x = self.pool1(x)

        x = self.flat(x)
        x = self.act3(self.fc(x))
        x = self.drop2(x)

        x = self.act4(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)

        return x