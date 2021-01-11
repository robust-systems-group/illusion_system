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
from torch_models import D2NN, Q
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

#ubit_8 = FloatingPoint(exp=5, man=2)
#ubit_16 = FloatingPoint(exp=6, man=9)

weight_quant = quantizer(forward_number=ubit_8,
                        forward_rounding="nearest")
grad_quant = quantizer(forward_number=None,
                        forward_rounding="nearest")
momentum_quant = quantizer(forward_number=None,
                        forward_rounding="stochastic")
acc_quant = quantizer(forward_number=None,
                        forward_rounding="stochastic")

act_error_quant = lambda : Quantizer(forward_number= ubit_8, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")

act2_error_quant = lambda : Quantizer(forward_number=ubit_16, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")


device = 'cuda' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

# state_dict = torch.load('../checkpoints/mnist_model.pth')
model = D2NN(act_error_quant, act2_error_quant)
# model.load_state_dict(state_dict)
model = model.to(device=device)
state_dict = torch.load('./checkpoints/mnist_d2nn_quant.pth')
model.load_state_dict(state_dict)

model_q = Q(act_error_quant, act2_error_quant)
model_q = model_q.to(device=device)

def run_epoch(loader, model, model_q, lamb, eps, criterion, optimizer=None, phase="train"):
    assert phase in ["train","eval"], "Invalid Phase"
    num_n2 = 0
    num_n3 = 0
    loss_sum_q = 0.0
    correct_q = 0.0
    loss_sum_n2 = 0.0
    correct_n2 = 0.0
    loss_sum_n3 = 0.0
    correct_n3 = 0.0

    if phase=="train":
        model_q.train()
        model.eval()
    elif phase=="eval":
        model_q.eval()
        model.eval()

    ttl = 0
    with torch.autograd.set_grad_enabled(phase=="train"):
        for i, (input, target) in tqdm(enumerate(loader), total = len(loader)):
            input = input.to(device=device)
            target = target.to(device=device)
             
            N1_out, N2_out, N3_out = model(input)
            Q_out = model_q(N1_out.detach())
            #print(Q_out) 
            r_mask = torch.rand_like(Q_out.detach()[:,0]) < eps
            N3_mask = torch.argmax(Q_out.detach(),dim=1).bool() != r_mask
            N2_mask = ~N3_mask
            
            N2_masked = N2_out.detach()[N2_mask,:]
            N3_masked = N3_out.detach()[N3_mask,:]
            target_N2 = target.detach()[N2_mask]
            target_N3 = target.detach()[N3_mask]
            
            Q_out_N2 = Q_out[N2_mask,0] 
            Q_out_N3 = Q_out[N3_mask,1] 
            
            Q_out.retain_grad() 
            Q_out_N2.retain_grad() 
            Q_out_N3.retain_grad() 
            
            if len(Q_out_N2) > 0:
                score_N2 = f1_score(target_N2.cpu(), torch.argmax(N2_masked,dim=1).cpu(), average='micro') 
                pred_n2 = N2_masked.data.argmax(1)
                score_N2 =  torch.sum(pred_n2.eq(target_N2)).float()/len(target_N2)
                #print(score_N2)
                cost_N2 = torch.Tensor([lamb*score_N2 + (1-lamb)*0.2]).to(device) #N1 is 5x cheaper
                #print(cost_N2)
                mean_Q_N2 = torch.mean(Q_out_N2,dim = 0,keepdim=True)
                mean_Q_N2.retain_grad()
                loss_N2 = criterion(mean_Q_N2,cost_N2)
                
                loss_sum_n2 += loss_N2.cpu().item()*input.size(0)
                num_n2 += len(Q_out_N2) 
                pred_n2 = N2_masked.data.argmax(1,keepdim=True)
                #score_N2 =  pred_n2.eq(target_N2.data.view_as(pred_n2)).sum().cpu().item()
                correct_n2 += pred_n2.eq(target_N2.data.view_as(pred_n2)).sum().cpu().item()
            else:
                loss_N2 = torch.FloatTensor([0]).to(device)
            
            if len(Q_out_N3) > 0:
                score_N3 = f1_score( target_N3.cpu(), torch.argmax(N3_masked,dim=1).cpu(), average='micro')
                pred_n3 = N3_masked.data.argmax(1)
                score_N3 =  torch.sum(pred_n3.eq(target_N3)).float()/len(target_N3)
                #print(score_N3)
                cost_N3 = torch.Tensor([lamb*score_N3 + (1-lamb)*1]).to(device) #N1 is 5x cheaper
                #print(cost_N3)
                mean_Q_N3 = torch.mean(Q_out_N3,dim=0,keepdim=True)
                mean_Q_N3.retain_grad()
                loss_N3 = criterion(mean_Q_N3, cost_N3)
                
                loss_sum_n3 += loss_N3.cpu().item()*input.size(0)
                num_n3 += len(Q_out_N3) 
                pred_n3 = N3_masked.data.argmax(1,keepdim=True)
                correct_n3 += pred_n3.eq(target_N3.data.view_as(pred_n3)).sum().cpu().item()
            else:
                loss_N3 = torch.FloatTensor([0]).to(device)
            
            if phase =="train":
                optimizer.zero_grad()
                loss_N2 = loss_N2*1000
                loss_N3 = loss_N3*1000
                (loss_N2 + loss_N3).backward()
                optimizer.step()
            #else:
            #   print(Q_out)
            
            ttl += input.size()[0]


    return {
        'percent_n2': num_n2 / float(ttl),
        'loss_n2': loss_sum_n2 / float(ttl),
        'accuracy_n2': np.float64(correct_n2) / np.float64(num_n2)*100,
        'percent_n3': num_n3 / float(ttl),
        'loss_n3': loss_sum_n3 / float(ttl),
        'accuracy_n3': np.float64(correct_n3) / np.float64(num_n3)*100
    }

optimizer = optim.SGD(model_q.parameters(), lr=0.000001, momentum=0.9)
optimizer = OptimLP(optimizer,
                    weight_quant=weight_quant,
                    grad_quant=grad_quant,
                    momentum_quant=momentum_quant,
                    acc_quant=acc_quant,
                    grad_scaling=1/1000)

exp_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.1)
eps = 0.5
lamb = 0.55

for epoch in range(20):
    eps = eps*0.9
    train_res = run_epoch(train_loader, model, model_q, lamb, eps, F.mse_loss, optimizer=optimizer, phase="train")
    print(train_res)
    test_res = run_epoch(test_loader, model, model_q, lamb, 0, F.mse_loss, optimizer=optimizer, phase="eval")
    print(test_res)
    exp_lr_scheduler.step()
    torch.save(model_q.state_dict(),'./checkpoints/mnist_d2nn_q_quant.pth')
