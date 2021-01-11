#
# Copyright (C) 2020 by The Board of Trustees of Stanford University
# This program is free software: you can redistribute it and/or modify it under
# the terms of the Modified BSD-3 License as published by the Open Source
# Initiative.
# If you use this program in your research, we request that you reference the
# Illusion paper, and that you send us a citation of your work.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the BSD-3 License for more details.
# You should have received a copy of the Modified BSD-3 License along with this
# program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
#


from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer,quantizer
from qtorch.optim import OptimLP

class MNIST_Net(nn.Module):
    def __init__(self, quant, quant2):
        super(MNIST_Net, self).__init__()

        self.conv1 = nn.Conv2d(1,6,5,1,0)
        self.conv2 = nn.Conv2d(6,6,3,1,0)
        self.conv3 = nn.Conv2d(6,8,3,1,0)
        self.conv4 = nn.Conv2d(8,24,3,1,0)
        self.fc = nn.Linear(24*2*2, 10)
        self.quant = quant()
        self.quant2 = quant2()

    def forward(self, x):
        x = self.quant(x)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,2,2)
        x = self.quant(x)
        
        x = F.relu(self.conv2(x))
        x = self.quant(x)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2,2)
        x = self.quant(x)
        
        x = F.relu(self.conv4(x))
        x = self.quant(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)
        x = self.quant2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class SVHN_Net(nn.Module):
    def __init__(self, quant, quant2):
        super(SVHN_Net, self).__init__()

        self.conv1 = nn.Conv2d(3,18,3,stride=1,padding=1)
        self.conv2 = nn.Conv2d(18,18,3,stride=1,padding=1)
        self.conv3 = nn.Conv2d(18,24,3,stride=1,padding=1)
        self.fc1 = nn.Linear(24*4*4, 52)
        self.fc2 = nn.Linear(52, 60)
        self.fc3 = nn.Linear(60, 10)
        self.quant = quant()
        self.quant2 = quant2()

    def forward(self, x):
        x = self.quant(x)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x,4,4)
        x = self.quant(x)
        
        x = F.relu(self.conv2(x))
        x = self.quant(x)
        
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x,2,2)
        x = self.quant(x)
        
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.quant2(x)
        
        x = self.fc2(x)
        x = self.quant2(x)
        
        x = self.fc3(x)
        x = self.quant2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class N1(nn.Module):
    def __init__(self, quant, quant2):
        super(N1, self).__init__()
        self.conv1 = nn.Conv2d(1,6,5,1,0)
        self.conv2 = nn.Conv2d(6,12,1,1,0)
        self.quant = quant()

    def forward(self, x):
        x = self.quant(x)
        
        x = F.relu(self.conv1(x))
        x = self.quant(x)

        x = F.max_pool2d(x,4)
        x = F.relu(self.conv2(x))
        x = self.quant(x)
        return x  


class Q(nn.Module):
    def __init__(self, quant, quant2):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(3*3*12, 64, bias = True)
        self.fc2 = nn.Linear(64, 2, bias = True)
        self.quant = quant()
        self.quant2 = quant2()

    def forward(self, x):
        x = self.quant(x)
        x = F.max_pool2d(x,kernel_size=2)
        x = x.view(-1, self.num_flat_features(x))
        
        x = self.fc1(x)
        x = self.quant2(x)
        
        x = self.fc2(x)
        x = self.quant2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class N2(nn.Module): # Large Network
    def __init__(self, quant, quant2):
        super(N2, self).__init__()
        self.conv1 = nn.Conv2d(12,12,3,1,0)
        self.conv2 = nn.Conv2d(12,24,3,1,0)
        self.fc1 = nn.Linear(2*2*24, 128)
        self.fc2 = nn.Linear(128, 24)
        self.fc3 = nn.Linear(24, 10)
        self.quant = quant()
        self.quant2 = quant2()

    def forward(self, x):
        x = self.quant(x)

        x = F.relu(self.conv1(x))
        x = self.quant(x)
        
        x = F.relu(self.conv2(x))
        x = self.quant(x)
        
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.quant2(x)
        
        x = self.fc2(x)
        x = self.quant2(x)
        
        x = self.fc3(x)
        x = self.quant2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class N3(nn.Module): # Small Network
    def __init__(self, quant, quant2):
        super(N3, self).__init__()
        self.fc1 = nn.Linear(3*3*12, 32)
        self.fc2 = nn.Linear(32, 10)
        self.quant = quant()
        self.quant2 = quant2()

    def forward(self, x):
        x = self.quant(x)
        x = F.max_pool2d(x,2)
        x = x.view(-1, self.num_flat_features(x))
        x = self.quant(x)
        
        x = self.fc1(x)
        x = self.quant2(x)
        
        x = self.fc2(x)
        x = self.quant2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class D2NN(nn.Module):
    def __init__(self, quant, quant2, use_q=False):
        super(D2NN, self).__init__()
        self.N1 = N1(quant, quant2)
        self.N2 = N2(quant, quant2)
        self.N3 = N3(quant, quant2)

    def forward(self,x):
        N1_out = self.N1(x)
        N2_out = self.N2(N1_out)
        N3_out = self.N3(N1_out)
        return N1_out, N2_out, N3_out

