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


from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
from sklearn.metrics import f1_score
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import random
import multiprocessing as mp
import pdb
from torch_models import D2NN
from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
import pdb
from tqdm import tqdm

best_result = 0
path = os.path.join("./data")
batchSize = 128

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=path, train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=batchSize, shuffle=True)

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root=path, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.5,), (0.5,))
                       ])),
        batch_size=batchSize, shuffle=True)

ubit_8 = FixedPoint(8, 4)
ubit_16 = FixedPoint(16, 8)

weight_quant = quantizer(forward_number=ubit_8,
                        forward_rounding="nearest")
grad_quant = quantizer(forward_number=None,
                        forward_rounding="nearest")
momentum_quant = quantizer(forward_number=None,
                        forward_rounding="stochastic")
acc_quant = quantizer(forward_number=None,
                        forward_rounding="stochastic")

act_error_quant = lambda : Quantizer(forward_number=ubit_8, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")

act2_error_quant = lambda : Quantizer(forward_number=ubit_16, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")


device = 'cuda' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# state_dict = torch.load('../checkpoints/mnist_model.pth')
model = D2NN(act_error_quant, act2_error_quant)
# model.load_state_dict(state_dict)
model = model.to(device=device)


def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    assert phase in ["train","eval"], "Invalid Phase"
    loss_sum_n2 = 0.0
    correct_n2 = 0.0
    loss_sum_n3 = 0.0
    correct_n3 = 0.0

    if phase=="train": model.train()
    elif phase=="eval":model.eval()

    ttl = 0
    with torch.autograd.set_grad_enabled(phase=="train"):
        for i, (input, target) in tqdm(enumerate(loader), total = len(loader)):
            input = input.to(device=device)
            target = target.to(device=device)
            N1_out, N2_out, N3_out = model(input)
            N2_loss = criterion(N2_out, F.one_hot(target, 10).float())
            N3_loss = criterion(N3_out, F.one_hot(target, 10).float())
            loss_sum_n2 += N2_loss.cpu().item()*input.size(0)
            loss_sum_n3 += N3_loss.cpu().item()*input.size(0)
            pred_n2 = N2_out.data.max(1,keepdim=True)[1]
            pred_n3 = N3_out.data.max(1,keepdim=True)[1]
            correct_n2 += pred_n2.eq(target.data.view_as(pred_n2)).sum()
            correct_n3 += pred_n3.eq(target.data.view_as(pred_n3)).sum()
            ttl += input.size()[0]

            if phase =="train":
                N2_loss = N2_loss*1000
                N3_loss = N3_loss*1000
                optimizer.zero_grad()
                (5*N2_loss+N3_loss).backward(retain_graph=True)
                optimizer.step()

    correct_n2 = correct_n2.cpu().item()
    correct_n3 = correct_n3.cpu().item()
    return {
        'loss_n2': loss_sum_n2 / float(ttl),
        'accuracy_n2': correct_n2 / float(ttl)*100.0,
        'loss_n3': loss_sum_n3 / float(ttl),
        'accuracy_n3': correct_n3 / float(ttl)*100.0
    }

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = OptimLP(optimizer,
                    weight_quant=weight_quant,
                    grad_quant=grad_quant,
                    momentum_quant=momentum_quant,
                    acc_quant=acc_quant,
                    grad_scaling=1/1000)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)

for epoch in range(20):
    train_res = run_epoch(train_loader, model, F.mse_loss, optimizer=optimizer, phase="train")
    print(train_res)
    test_res = run_epoch(test_loader, model, F.mse_loss, optimizer=optimizer, phase="eval")
    print(test_res)

torch.save(model.state_dict(), './checkpoints/mnist_d2nn_quant.pth')
