import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SkinModel(nn.Module):
    def __init__(self):
        super(SkinModel, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 7)  
        self.dropout = nn.Dropout(0.4) 

    def forward(self, x):
        x = self.resnet(x)
        x = self.dropout(x) 
        return x 
