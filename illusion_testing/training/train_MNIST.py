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
from torchvision.datasets import ImageFolder
import os
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.autograd import Variable
from qtorch import FixedPoint, FloatingPoint
from qtorch.quant import Quantizer, quantizer
from qtorch.optim import OptimLP
import pdb
from tqdm import tqdm
import torch_models

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
model = torch_models.MNIST_Net(act_error_quant, act2_error_quant)
# model.load_state_dict(state_dict)
model = model.to(device=device)




def run_epoch(loader, model, criterion, optimizer=None, phase="train"):
    assert phase in ["train","eval"], "Invalid Phase"
    loss_sum = 0.0
    correct = 0.0

    if phase=="train": model.train()
    elif phase=="eval":model.eval()

    ttl = 0
    with torch.autograd.set_grad_enabled(phase=="train"):
        for i, (input, target) in tqdm(enumerate(loader), total = len(loader)):
            input = input.to(device=device)
            target = target.to(device=device)
            output = model(input)
            loss = criterion(output,target)
            loss_sum += loss.cpu().item()*input.size(0)
            pred = output.data.max(1,keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            ttl += input.size()[0]

            if phase =="train":
                loss = loss*1000
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    correct = correct.cpu().item()
    return {
        'loss': loss_sum / float(ttl),
        'accuracy': correct / float(ttl)*100.0
    }
schedule = [0.1,0.01,0.001]

for epoch in range(30):
    lr = schedule[int(epoch/10)]
    #train(model, device, train_loader, optimizer, epoch)
    #test(model, device, test_loader, epoch)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    optimizer = OptimLP(optimizer,
                    weight_quant=weight_quant,
                    grad_quant=grad_quant,
                    momentum_quant=momentum_quant,
                    acc_quant=acc_quant,
                    grad_scaling=1/1000)
    train_res = run_epoch(train_loader, model, F.cross_entropy, optimizer=optimizer, phase="train")
    print(train_res)
    test_res = run_epoch(test_loader, model, F.cross_entropy, optimizer=optimizer, phase="eval")
    print(test_res)
    # Do checkpointing - Is saved in outf
torch.save(model.state_dict(), './checkpoints/mnist_model_quant2.pth')


