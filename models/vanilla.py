'''ConvNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class VanBlock(nn.Module):
    def __init__(self,in_channel,out_channel,kernel,padding):
        super(VanBlock, self).__init__()
        self.subblock=nn.Sequential(
            nn.Conv2d(in_channel,out_channel,kernel_size=kernel,padding=padding),
            nn.ReLU()
        )
    def forward(self,x):
        return self.subblock(x)

class Vanilla(nn.Module):
    def __init__(self, blocknum,num_classes=10):
        super(Vanilla, self).__init__()
        list=[]
        list.append(VanBlock(3,32,3,1))
        for i in range(blocknum-1):
            list.append(VanBlock(32,32,3,1))
        self.conv=nn.Sequential(*list)
        # self.drop=nn.Dropout()
        self.fc1   = nn.Linear(32*32*32, num_classes)
    def forward(self, x):
        out=self.conv(x)
        out = out.view(out.size(0), -1)
        # out=self.drop(out)
        out = self.fc1(out)
        return out
def Vanilla3(num_classes):
    return Vanilla(3,num_classes=num_classes)

def Vanilla8(num_classes):
    return Vanilla(8,num_classes=num_classes)

def Vanilla13(num_classes):
    return Vanilla(13,num_classes=num_classes)

def Vanilla18(num_classes):
    return Vanilla(18,num_classes=num_classes)
