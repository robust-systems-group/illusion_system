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
from qtorch.quant import Quantizer,quantizer
from qtorch.optim import OptimLP
import pdb
from tqdm import tqdm
import torch_models

best_result = 0
batchSize = 128

test_data = np.load('data_MNIST.npz')
print(test_data['data'].shape)
data = torch.from_numpy(test_data['data'])
target = torch.from_numpy(test_data['labels'])

ubit_8 = FixedPoint(8, 4)
ubit_16 = FixedPoint(16, 8)

weight_quant = quantizer(forward_number=ubit_8, forward_rounding="nearest")
grad_quant = quantizer(forward_number=None, forward_rounding="nearest")
momentum_quant = quantizer(forward_number=None, forward_rounding="stochastic")
acc_quant = quantizer(forward_number=None, forward_rounding="stochastic")

act_error_quant = lambda : Quantizer(forward_number=ubit_8, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")

act2_error_quant = lambda : Quantizer(forward_number=ubit_16, backward_number=None,
                        forward_rounding="nearest", backward_rounding="stochastic")

device = 'cpu' # torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch_models.D2NN(act_error_quant, act2_error_quant)
model.load_state_dict(torch.load('./checkpoints/mnist_d2nn_quant.pth'))
model = model.to(device=device)
model.eval()


activation = {}
def get_activation(name, side):
    def hook(model, input, output):
        if side==1: activation[name] = output#input#.detach()
        else: activation[name] = input#input#.detach()
    return hook

model.N1.conv1.register_forward_hook(get_activation('n1_conv1_in',0))
model.N1.conv1.register_forward_hook(get_activation('n1_conv1_out',1))
model.N1.conv2.register_forward_hook(get_activation('n1_conv2_in',0))
model.N1.conv2.register_forward_hook(get_activation('n1_conv2_out',1))

model.N2.conv1.register_forward_hook(get_activation('n2_conv1_in',0))
model.N2.conv1.register_forward_hook(get_activation('n2_conv1_out',1))
model.N2.conv2.register_forward_hook(get_activation('n2_conv2_in',0))
model.N2.conv2.register_forward_hook(get_activation('n2_conv2_out',1))
model.N2.fc1.register_forward_hook(get_activation('n2_fc1_in',0))
model.N2.fc1.register_forward_hook(get_activation('n2_fc1_out',1))
model.N2.fc2.register_forward_hook(get_activation('n2_fc2_in',0))
model.N2.fc2.register_forward_hook(get_activation('n2_fc2_out',1))

model.N3.fc1.register_forward_hook(get_activation('n3_fc1_in',0))
model.N3.fc1.register_forward_hook(get_activation('n3_fc1_out',1))
model.N3.fc2.register_forward_hook(get_activation('n3_fc2_in',0))
model.N3.fc2.register_forward_hook(get_activation('n3_fc2_out',1))



model_q = torch_models.Q(act_error_quant, act2_error_quant)
model_q.load_state_dict(torch.load('./checkpoints/mnist_d2nn_q_quant.pth'))
model_q = model_q.to(device=device)
model_q.eval()

model_q.fc1.register_forward_hook(get_activation('q_fc1_in',0))
model_q.fc1.register_forward_hook(get_activation('q_fc1_out',1))
model_q.fc2.register_forward_hook(get_activation('q_fc2_in',0))
model_q.fc2.register_forward_hook(get_activation('q_fc2_out',1))


n1_out, n2_out, n3_out  = model(data)
q_out = model_q(n1_out)


pred_q = [np.argmax(i).item() for i in q_out.detach()]
pred = [np.argmax(n2_out.detach()[i]).item() if not pred_q[i] else np.argmax(n3_out.detach()[i]).item() for i in range(len(pred_q))]
pred_n2 = [np.argmax(i).item() for i in n2_out.detach()]
pred_n3 = [np.argmax(i).item() for i in n3_out.detach()]

np.savez('int_MNIST_D2NN',
        n1_conv1_in =      activation['n1_conv1_in'][0].detach(), 
        n1_conv1_out =     activation['n1_conv1_out'][0].detach(), 
        n1_conv2_in =      activation['n1_conv2_in'][0].detach(), 
        n1_conv2_out =     activation['n1_conv2_out'][0].detach(), 
        
        n2_conv1_in =      activation['n2_conv1_in'][0].detach(), 
        n2_conv1_out =     activation['n2_conv1_out'][0].detach(), 
        n2_conv2_in =      activation['n2_conv2_in'][0].detach(), 
        n2_conv2_out =     activation['n2_conv2_out'][0].detach(), 
        n2_fc1_in =      activation['n2_fc1_in'][0].detach(), 
        n2_fc1_out =     activation['n2_fc1_out'][0].detach(), 
        n2_fc2_in =      activation['n2_fc2_in'][0].detach(), 
        n2_fc2_out =     activation['n2_fc2_out'][0].detach(), 

        n3_fc1_in =      activation['n3_fc1_in'][0].detach(), 
        n3_fc1_out =     activation['n3_fc1_out'][0].detach(), 
        n3_fc2_in =      activation['n3_fc2_in'][0].detach(), 
        n3_fc2_out =     activation['n3_fc2_out'][0].detach(),

        q_fc1_in =      activation['q_fc1_in'][0].detach(), 
        q_fc1_out =     activation['q_fc1_out'][0].detach(), 
        q_fc2_in =      activation['q_fc2_in'][0].detach(), 
        q_fc2_out =     activation['q_fc2_out'][0].detach(),
        which_q =       pred_q,
        )

print(np.sum(np.array(pred) == target.numpy())/1000)
print(np.sum(np.array(pred_n2) == target.numpy())/1000)
print(np.sum(np.array(pred_n3) == target.numpy())/1000)
print(np.sum(np.array(pred_q))/1000)
