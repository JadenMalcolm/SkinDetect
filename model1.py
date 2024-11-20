import torch.nn as nn
import torch.nn.functional as F

class SkinModel(nn.Module):
    def __init__(self):
        super(SkinModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool_last = nn.MaxPool2d(2, 1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 2 * 2, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool_last(F.relu(self.conv4(x)))
        x = self.flatten(x)
        x = self.fc1(x)  
        return x
