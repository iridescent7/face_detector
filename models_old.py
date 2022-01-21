import torch
import torch.nn as nn

class PNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fcn = nn.ModuleList([
            nn.Conv2d(3, 10, kernel_size=3),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.PReLU(10),
            nn.Conv2d(10, 16, kernel_size=3),
            nn.PReLU(16),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.PReLU(32)
        ])

        self.fc_conv = nn.Conv2d(32, 2, kernel_size=1)
        self.fc_softmax = nn.Softmax(dim=1)

        self.bb_conv = nn.Conv2d(32, 4, kernel_size=1)
    
    def forward(self, x):
        for l in self.fcn:
            x = l(x)
        
        fc = self.fc_softmax(self.fc_conv(x))
        bb = self.bb_conv(x)

        return fc, bb

class RNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fcn = nn.ModuleList([
            nn.Conv2d(3, 28, kernel_size=3),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.PReLU(28),
            nn.Conv2d(28, 48, kernel_size=3),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.PReLU(48),
            nn.Conv2d(48, 64, kernel_size=2),
            nn.PReLU(64)
        ])

        self.fcl = nn.ModuleList([
            nn.Linear(1024, 128),
            nn.PReLU(128)
        ])

        self.fc_dense = nn.Linear(128, 2)
        self.fc_softmax = nn.Softmax(dim=1)

        self.bb_dense = nn.Linear(128, 4)
    
    def forward(self, x):
        for l in self.fcn:
            x = l(x)
        
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(x.shape[0], -1)

        for l in self.fcl:
            x = l(x)

        fc = self.fc_softmax(self.fc_dense(x))
        bb = self.bb_dense(x)

        return fc, bb

class ONet(nn.Module):
    def __init__(self):
        super().__init__()

        self.fcn = nn.ModuleList([
            nn.Conv2d(3, 32, kernel_size=3),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.PReLU(32),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=2),
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.PReLU(64),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.PReLU(128)
        ])

        self.fcl = nn.ModuleList([
            nn.Linear(2048, 256),
            nn.PReLU(256)
        ])

        self.fc_dense = nn.Linear(256, 2)
        self.fc_softmax = nn.Softmax(dim=1)

        self.bb_dense = nn.Linear(256, 4)
        # self.fl_dense = nn.Linear(256, 10)
    
    def forward(self, x):
        for l in self.fcn:
            x = l(x)
        
        x = x.permute(0, 3, 2, 1).contiguous()
        x = x.view(x.shape[0], -1)

        for l in self.fcl:
            x = l(x)

        fc = self.fc_softmax(self.fc_dense(x))
        bb = self.bb_dense(x)
        # fl = self.fl_dense(x)

        return fc, bb