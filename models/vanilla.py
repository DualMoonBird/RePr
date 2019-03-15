'''ConvNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class Vanilla(nn.Module):
    def __init__(self, num_classes=10):
        super(Vanilla, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1   = nn.Linear(32*4*4, num_classes)

    def forward(self, x):
        out = self.maxpool(F.relu(self.conv1(x)))
        out = self.maxpool(F.relu(self.conv2(out)))
        out = self.maxpool(F.relu(self.conv3(out)))
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out
